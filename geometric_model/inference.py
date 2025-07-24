import argparse
import torch
import numpy as np
import nibabel as nib

from deep_sdf_decoder import Decoder
from prepare_sdf_samples import compute_target_sdf, sample_sdf_balanced

def load_segmentation(path):
    nii = nib.load(path)
    arr = nii.get_fdata() > 0
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D], CPU

def reconstruct_latent(decoder, init_z, coords, sdf_vals, **opt_kwargs):
    """
    Given sampled coords (Nx3) and sdf_vals (N,), optimize latent z
    """
    return reconstruct(decoder, init_z, (coords, sdf_vals), **opt_kwargs)

def decode_full_sdf_grid(decoder, z, shape):
    """
    Build a full [-1,1]^3 grid matching `shape=(H,W,D)`, decode
    SDF at each voxel, and return as numpy array [H,W,D].
    """
    H, W, D = shape
    xs = np.linspace(-1, 1, H)
    ys = np.linspace(-1, 1, W)
    zs = np.linspace(-1, 1, D)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)  # [H,W,D,3]
    flat = torch.from_numpy(grid.reshape(-1,3).astype(np.float32)).cuda()
    latent_exp = z.expand(flat.shape[0], -1)
    with torch.no_grad():
        preds = decoder(torch.cat([latent_exp, flat], dim=1)).cpu().numpy()
    return preds.reshape(H, W, D)

def save_nifti(arr, ref_nii, out_path):
    out = nib.Nifti1Image(arr.astype(np.float32), ref_nii.affine, ref_nii.header)
    nib.save(out, out_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--segmentation", required=True,
                        help="Your nnUNet output (imperfect) NIfTI")
    parser.add_argument("--output_uncertainty", default="uncertainty.nii.gz")
    parser.add_argument("--latent_size", type=int, default=256)
    parser.add_argument("--hidden_dims", type=str,
                        default="512,512,512,512,512,512,512,512")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    # 1) Load decoder
    ckpt = torch.load(args.checkpoint, map_location=device)
    decoder = Decoder(
        latent_size=args.latent_size,
        dims=hidden_dims,
        dropout=list(range(len(hidden_dims))),
        dropout_prob=0.2,
        norm_layers=list(range(len(hidden_dims))),
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()

    # 2) Initialize a random latent (or zero)
    z = torch.zeros(1, args.latent_size, device=device).normal_(0, 0.01)

    # 3) Load your imperfect seg → compute SDF field
    ref_nii = nib.load(args.segmentation)
    seg_t = load_segmentation(args.segmentation).to(device)
    sdf_t = compute_target_sdf(seg_t, clip_threshold=10).to(device)  # [1,1,H,W,D]
    sdf_np = sdf_t.squeeze().cpu().numpy()  # [H,W,D]

    # 4) Sample surface + noisy points
    coords, sdf_vals = sample_sdf_balanced(seg_t, sdf_t,
                                           num_points=args.num_samples,
                                           noise_std=args.noise_std)
    coords = torch.from_numpy(coords).float().to(device)    # [N,3]
    sdf_vals = torch.from_numpy(sdf_vals).float().unsqueeze(1).to(device)  # [N,1]

    # 5) Optimize latent to fit that SDF
    from reconstruct_mesh import reconstruct  # or wherever you defined it
    z_opt = reconstruct(decoder, z, (coords, sdf_vals),
                        num_iterations=args.iters, clamp_dist=10.0, lr=5e-3, l2reg=True)

    # 6) Decode a full SDF grid at original volume resolution
    H, W, D = sdf_np.shape
    pred_sdf = decode_full_sdf_grid(decoder, z_opt, (H, W, D))

    # 7) Compute uncertainty = |predicted − input|
    uncertainty = np.abs(pred_sdf - sdf_np)

    # 8) Save as NIfTI (aligned with your input)
    save_nifti(uncertainty, ref_nii, args.output_uncertainty)
    print("Wrote uncertainty map to", args.output_uncertainty)
