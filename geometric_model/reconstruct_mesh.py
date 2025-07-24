import argparse
import torch
import torch.nn as nn
from deep_sdf_decoder import Decoder
import numpy as np
import nibabel as nib

from from_DeepSDF.mesh import create_mesh
from prepare_sdf_samples import compute_target_sdf, sample_sdf_balanced

def optimize_latent(decoder, init_z, seg_t, num_surface=5000, num_interior=5000, num_background=5000, noise_std=0.3,
                    num_iters=1000, lr=5e-3, clamp_dist=10.0, lambda_prior=1e-2):
    """
    Args:
      - decoder: your loaded Decoder
      - init_z: tensor [1, latent_size], initial guess (e.g. zero)
      - seg_t: tensor [1,1,H,W,D] of imperfect segmentation mask
    Runs gradient descent on z to fit decoder(z, coords) ≈ seg_t,
    with additional interior/background constraints and latent prior.
    Returns:
        Optimized latent code tensor z of shape [1, latent_size].
    """
    device = init_z.device
    _, _, H, W, D = seg_t.shape

    # 1) Compute full SDF volume from the imperfect segmentation
    sdf_vol = compute_target_sdf(seg_t, clip_threshold=clamp_dist).squeeze().cpu().numpy()
    sdf_vol = torch.from_numpy(sdf_vol).unsqueeze(1).to(device)  # [1,1,H,W,D]

    # 2) Sample your three sets of points
    # 2a) Surface band + noisy
    coords_s, sdf_s = sample_sdf_balanced(sdf_vol, num_points=num_surface, noise_std=noise_std)
    coords_s = torch.from_numpy(coords_s).float().to(device)         # [num_surface,3]
    sdf_s = torch.from_numpy(sdf_s).float().unsqueeze(1).to(device)  # [num_surface,1]

    # 2b) Build a flat list of all voxel indices and mask for interior/background
    coords_grid = np.argwhere(np.ones((H,W,D))).astype(np.float32)   # [H*W*D,3]
    seg_flat = seg_t.squeeze().cpu().numpy().reshape(-1)             # [H*W*D]
    interior = coords_grid[seg_flat > 0]
    background = coords_grid[seg_flat == 0]

    # helper to normalize voxel coords to [-1,1]
    def normalize(pts):
        pts = pts.copy()
        pts[:,0] = pts[:,0]/(H-1)*2 - 1
        pts[:,1] = pts[:,1]/(W-1)*2 - 1
        pts[:,2] = pts[:,2]/(D-1)*2 - 1
        return pts

    # 2c) Random interior points (enforce negative SDF)
    idx_int = np.random.choice(len(interior), num_interior, replace=False)
    pts_int_norm = normalize(interior[idx_int])
    pts_int = torch.from_numpy(pts_int_norm).float().to(device)       # [num_interior,3]
    sdf_int = -torch.abs(torch.randn(num_interior, 1, device=device))

    # 2d) Random background points (enforce positive SDF)
    idx_bg = np.random.choice(len(background), num_background, replace=False)
    pts_bg_norm = normalize(background[idx_bg])
    pts_bg = torch.from_numpy(pts_bg_norm).float().to(device)         # [num_background,3]
    sdf_bg = torch.abs(torch.randn(num_background, 1, device=device))

    # 3) Concatenate once up-front
    coords_all = torch.cat([coords_s, pts_int, pts_bg], dim=0)  # [(num_surf+num_int+num_bg), 3]
    sdf_all = torch.cat([sdf_s, sdf_int, sdf_bg], dim=0)

    # 4) Set up latent optimization
    z = init_z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    loss_fn = nn.L1Loss(reduction="mean")

    # 5) Optimization loop uses the fixed coords_all / sdf_all
    for i in range(num_iters):
        optimizer.zero_grad()

        z_exp = z.expand(coords_all.shape[0], -1)        # [batch, latent_size]
        preds = decoder(torch.cat([z_exp, coords_all], 1))
        preds = torch.clamp(preds, -clamp_dist, clamp_dist)

        loss_data = loss_fn(preds, sdf_all)
        loss_reg = lambda_prior * torch.mean(z.pow(2))
        (loss_data + loss_reg).backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iter {i:4d}: data_loss={loss_data:.4f}, prior={loss_reg:.4f}")

    return z.detach()


def load_model_and_latents(checkpoint_path, latent_size, hidden_dims, device):
    # 1) Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 2) Build decoder and load weights
    decoder = Decoder(
        latent_size=latent_size,
        dims=hidden_dims,
        dropout=[i for i in range(len(hidden_dims))],
        dropout_prob=0.2,
        norm_layers=[i for i in range(len(hidden_dims))],
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()

    # 3) Build embedding for latents and load checkpoint
    num_shapes = ckpt["latents_state"]["weight"].shape[0]
    lat_vecs = torch.nn.Embedding(num_shapes, latent_size).to(device)
    lat_vecs.load_state_dict(ckpt["latents_state"])
    lat_vecs.eval()

    return decoder, lat_vecs

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a mesh directly from a saved DeepSDF decoder + latents."
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to your saved auto_decoder.pth (or .pth)."
    )
    parser.add_argument(
        "--shape_idx", "-i", type=int,
        help="Index of the shape to reconstruct (0-based, from your training mapping)."
    )
    parser.add_argument(
        "--segmentation", "-s",
        help="If provided, path to an *imperfect* seg NIfTI to optimize for."
    )
    parser.add_argument(
        "--output", "-o", default="reconstructed_shape.ply",
        help="Filename for the output .ply mesh."
    )
    parser.add_argument(
        "--latent_size", type=int, default=256,
        help="Dimensionality of the latent code (must match training)."
    )
    parser.add_argument(
        "--hidden_dims", type=str,
        default="512,512,512,512,512,512,512,512",
        help="Comma‑separated list of decoder hidden dimensions."
    )
    parser.add_argument(
        "--resolution", "-N", type=int, default=256,
        help="Grid resolution for marching cubes (N x N x N)."
    )
    parser.add_argument(
        "--max_batch", type=int, default=32**3,
        help="Max #points per forward pass (useful to avoid OOM)."
    )
    parser.add_argument(
        "--no_xyz_in_all", action="store_true",
        help="If set, disables xyz_in_all in the decoder instantiation."
    )
    parser.add_argument(
        "--l2_penalty", type=float, default=1e-1,
        help="Loss penalty penalizing the deviation of the network prediction from the actual SDF value. Higher values \
        will hold z closer to 0 -> closer to the center of the ten learned vectors."
    )
    parser.add_argument(
        "--opt_num_iters", type=int, default=2500,
        help="Number of iterations the optimizer runs for."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    # Load model + latents
    decoder, lat_vecs = load_model_and_latents(
        args.checkpoint, args.latent_size, hidden_dims, device
    )
    
    if args.segmentation:
        # UNKOWN SHAPE: optimize latent to match this seg
        # load & compute SDF
        nii = nib.load(args.segmentation)
        arr = (nii.get_fdata() > 0).astype(np.float32)
        seg_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W,D]

        # init latent as zero
        init_z = torch.zeros(1, args.latent_size, device=device)
        z_opt = optimize_latent(decoder, init_z, seg_t,
                                num_iters=args.opt_num_iters, lr=5e-3, clamp_dist=10.0, lambda_prior=args.l2_penalty)
        latent_code = z_opt
        # compute L2 distances to all pre-trained latents
        all_latents = lat_vecs.weight.data  # [num_shapes, latent_size]
        dists = torch.norm(all_latents - latent_code, dim=1)  # [num_shapes]
        best_idx = torch.argmin(dists).item()
        print(f"Optimized latent is closest to pre-trained latent #{best_idx} "
              f"(distance={dists[best_idx]:.4f})")

    else:
        # KNOWN SHAPE: look up from embedding
        assert args.shape_idx is not None, "Either --shape_idx or --segmentation required"
        idx = torch.tensor([args.shape_idx], dtype=torch.long, device=device)
        latent_code = lat_vecs(idx)     # shape [1, latent_size]

    # Create the mesh
    base, _ = args.output.rsplit(".", 1)
    create_mesh(
        decoder,
        latent_code,
        filename=base,
        N=args.resolution,
        max_batch=args.max_batch,
    )
    print(f"Mesh saved as {base}.ply")

if __name__ == "__main__":
    main()
