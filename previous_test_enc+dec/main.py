import argparse
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from tqdm import tqdm
import math

from geometric_model.geometric_model import (
    GeometricLatentSetModel, 
    compute_target_sdf, 
    visualize_uncertainty_3d,
    save_reconstructed_mesh,
    sample_coords
)
from refinement_unet.refinement_unet import BasicUNetRefiner
from geometric_model.utils import (
    load_nifti_as_tensor,
    plot_loss_curves,
    visualize_sdf_predictions,
    plot_latent_distribution,
    save_full_sdf_nifti,
    save_full_sdf_nifti_wrapper,
    reconstruct_full_sdf_volume
)
from geometric_model.create_synthetic_mask import (
    create_synthetic_mask,
    save_synthetic_mask_nifti
)

# def save_reconstructed_segmentation(geo_model, seg_mask, optimized_latent_set, output_resolution=(64,64,64), save_path="reconstructed_segmentation.nii.gz"):
#     """
#     Generates a dense grid of coordinates over the volume defined by seg_mask,
#     uses the decoder with the optimized latent set to predict SDF values,
#     thresholds at zero to obtain a binary segmentation, and saves the full 3D volume
#     as a NIfTI file.
    
#     Args:
#       geo_model: GeometricLatentSetModel instance.
#       seg_mask: tensor of shape [1, 1, H, W, D] (imperfect segmentation).
#       optimized_latent_set: tensor of shape [1, M, latent_dim].
#       output_resolution: tuple (H_r, W_r, D_r) for the reconstruction grid.
#       save_path: file path for saving the NIfTI file.
#     """
#     B, C, H, W, D = seg_mask.shape
#     H_r, W_r, D_r = output_resolution
#     xs = torch.linspace(0, H-1, H_r)
#     ys = torch.linspace(0, W-1, W_r)
#     zs = torch.linspace(0, D-1, D_r)
#     grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)  # [H_r, W_r, D_r, 3]
#     grid = grid.view(1, -1, 3).to(seg_mask.device)  # [1, num_points, 3]
    
#     sdf_pred, _ = geo_model.decoder(optimized_latent_set, grid)
#     sdf_pred = sdf_pred.view(H_r, W_r, D_r).detach().cpu().numpy()
#     reconstructed = (sdf_pred < 0).astype(np.uint8)
#     affine = np.eye(4)
#     nii_img = nib.Nifti1Image(reconstructed, affine)
#     nib.save(nii_img, save_path)
#     print(f"Reconstructed segmentation saved as {save_path}")

# def visualize_full_sdf_volume(sdf_volume, n_cols=8, save_path=None):
#     """
#     Visualizes the full SDF volume by displaying each depth slice in a grid.
    
#     Args:
#       sdf_volume: Torch tensor or numpy array of shape [H, W, D].
#       n_cols: Number of columns in the grid.
#       save_path: If provided, saves the plot; otherwise displays it.
#     """
#     if torch.is_tensor(sdf_volume):
#         sdf_volume = sdf_volume.detach().cpu().numpy()
    
#     H, W, D = sdf_volume.shape
#     n_rows = math.ceil(D / n_cols)
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
#     axes = np.array(axes)
    
#     for idx in range(n_rows * n_cols):
#         ax = axes.flat[idx]
#         if idx < D:
#             slice_img = sdf_volume[:, :, idx]
#             im = ax.imshow(slice_img, cmap="jet")
#             ax.set_title(f"Slice {idx}")
#             ax.axis("off")
#         else:
#             ax.axis("off")
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         plt.close(fig)
#         print(f"Full SDF volume visualization saved to {save_path}")
#     else:
#         plt.show()

#############################################
# Training, Inference, and Visualization Functions
#############################################
def train_geometric_model(geo_model, seg_paths, device='cuda', epochs=5, lr=1e-4, num_points=2000):
    optimizer = optim.Adam(geo_model.parameters(), lr=lr)
    recon_losses = []
    kl_losses = []
    total_losses = []
    for epoch in range(epochs):
        start_time = time.time()
        total_loss_epoch = 0.0
        total_recon = 0.0
        total_kl = 0.0
        for seg_path in tqdm(seg_paths, desc=f"Epoch {epoch+1}/{epochs}", unit="case"):
            seg_vol = load_nifti_as_tensor(seg_path).to(device)
            _, _, h, w, d = seg_vol.shape
            coords = sample_coords(seg_vol, num_points, device=device, boundary_threshold=5)
            sdf_pred, sdf_loss, kl_loss = geo_model.forward_with_loss(seg_vol, coords)
            loss = sdf_loss + 1e-3 * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_recon += sdf_loss.item()
            total_kl += kl_loss.item()
        epoch_time = time.time() - start_time
        avg_loss = total_loss_epoch / len(seg_paths)
        avg_recon = total_recon / len(seg_paths)
        avg_kl = total_kl / len(seg_paths)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        total_losses.append(avg_loss)
        remaining_epochs = epochs - (epoch + 1)
        est_remaining = epoch_time * remaining_epochs
        print(f"[Epoch {epoch+1}/{epochs}] Loss = {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}) | Time: {epoch_time:.2f}s, Estimated remaining: {est_remaining:.2f}s")
    epochs_list = list(range(1, epochs+1))
    plot_loss_curves(epochs_list, recon_losses, kl_losses, total_losses, save_path="loss_curves.png")
    print("Finished training the geometric model! Loss curves saved as 'loss_curves.png'.")
    torch.save(geo_model.state_dict(), "geo_model.pth")
    print("Geometric model weights saved as 'geo_model.pth'.")

def train_refinement_unet(refiner, ct_paths, seg_paths, device='cuda', epochs=5, lr=1e-4):
    optimizer = optim.Adam(refiner.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for ct_path, seg_path in zip(ct_paths, seg_paths):
            ct_vol = load_nifti_as_tensor(ct_path).to(device)
            seg_vol = load_nifti_as_tensor(seg_path).to(device)
            uncertainty_map = torch.randn_like(seg_vol)
            refiner_input = torch.cat([ct_vol, seg_vol, uncertainty_map], dim=1)
            refined_seg = refiner(refiner_input)
            loss = criterion(refined_seg, seg_vol)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(ct_paths)
        print(f"[Refiner][Epoch {epoch+1}/{epochs}] Loss = {avg_loss:.4f}")
    print("Finished training the refinement UNet!")

def inference_dummy(geo_model, refiner, new_seg_path, new_ct_path=None, recon_resolution=(128,128,128), device='cuda'):
    seg_vol = load_nifti_as_tensor(new_seg_path).to(device)
    B, C, H, W, D = seg_vol.shape
    coords = sample_coords(seg_vol, num_points=3000, device=device, boundary_threshold=5)
    sdf_pred, sdf_loss, kl_loss = geo_model.forward_with_loss(seg_vol, coords)
    visualize_sdf_predictions(seg_vol, sdf_pred, coords, save_path="sdf_comparison.png")
    print("Saved SDF comparison plot as 'sdf_comparison.png'")
    latent_set = geo_model.encoder(seg_vol)
    plot_latent_distribution(latent_set, save_path="latent_distribution.png")
    print("Saved latent distribution plot as 'latent_distribution.png'")
    
    # Generate a full SDF volume from the GT segmentation using the encoder's latent set.
    full_sdf = reconstruct_full_sdf_volume(geo_model, seg_vol, latent_set, output_resolution=recon_resolution)
    save_full_sdf_nifti(full_sdf, save_path="full_sdf_volume.nii.gz")
    print("Saved full SDF volume as 'full_sdf_volume.nii.gz'")
    
    # Additionally, save a mesh from this full SDF volume.
    save_reconstructed_mesh(geo_model, seg_vol, latent_set, output_resolution=recon_resolution, save_path="reconstructed_mesh.obj")
    
    if new_ct_path is not None:
        ct_vol = load_nifti_as_tensor(new_ct_path).to(device)
        uncertainty_map = torch.randn_like(seg_vol)
        refiner_input = torch.cat([ct_vol, seg_vol, uncertainty_map], dim=1)
        refined_seg = refiner(refiner_input)
        print("Refinement output shape:", refined_seg.shape)
        return refined_seg
    print("Skipping refinement since no CT was provided.")
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Train/inference geometric and refinement models; geometric model uses segmentation masks only with latent set representation."
    )
    parser.add_argument('--data_folder_seg', type=str, required=True,
                        help="Path to folder containing .nii.gz segmentation volumes.")
    parser.add_argument('--data_folder_ct', type=str, required=False,
                        help="(Optional) Path to folder containing .nii.gz CT volumes.")
    parser.add_argument('--infer_folder', type=str, default=None,
                        help="(Optional) Path to folder containing imperfect segmentation files for inference.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda', 'cpu', or 'mps'.")
    parser.add_argument('--epochs_geo', type=int, default=5)
    parser.add_argument('--epochs_refiner', type=int, default=5)
    parser.add_argument('--lr_geo', type=float, default=1e-4)
    parser.add_argument('--lr_refiner', type=float, default=1e-4)
    parser.add_argument('--do_refiner', action='store_true',
                        help="If set, also train the refinement UNet using CT data.")
    parser.add_argument('--load_weights', type=str, default=None,
                        help="Path to saved geometric model weights to load for inference.")
    parser.add_argument('--mesh_resolution', type=str, default="128,128,128",
                        help="Comma-separated grid resolution for mesh reconstruction, e.g. '128,128,128'.")
    parser.add_argument('--create_synthetic', action='store_true',
                        help="If set, create a synthetic segmentation mask and save it as synthetic_mask.nii.gz for testing.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # If requested, create and save a synthetic segmentation mask.
    if args.create_synthetic:
        synthetic_mask = create_synthetic_mask(shape=(128,128,128), sphere_radius=40)
        save_synthetic_mask_nifti(synthetic_mask, save_path="synthetic_mask.nii.gz")
        # Update the segmentation folder to current directory for testing.
        seg_file = "synthetic_mask.nii.gz"
        print("Using synthetic segmentation mask for testing.")
    else:
        # 1) Gather ground truth segmentation file paths.
        seg_files = sorted(glob.glob(os.path.join(args.data_folder_seg, "topcow_ct_***.nii.gz")))
        if len(seg_files) == 0:
            print("No segmentation files found. Please check your data folder.")
            return
        seg_file = seg_files[0]
        print(f"Using segmentation file: {seg_file}")
    
    seg_vol = load_nifti_as_tensor(seg_file)
    input_size = seg_vol.shape[2:]
    print(f"Inferred segmentation volume size: {input_size}")

    mesh_res = tuple(int(x.strip()) for x in args.mesh_resolution.split(','))

    # 2) Initialize the geometric model.
    geo_model = GeometricLatentSetModel(latent_dim=64, M=128, input_size=input_size, hidden_dim=64, decoder_type='mlp').to(device)
    if args.load_weights is not None:
        geo_model.load_state_dict(torch.load(args.load_weights))
        print(f"Loaded geometric model weights from {args.load_weights}")
    else:
        # Train using segmentation folder.
        seg_files = sorted(glob.glob(os.path.join(args.data_folder_seg, "topcow_ct_***.nii.gz")))
        train_geometric_model(geo_model, seg_paths=seg_files, device=device, epochs=args.epochs_geo, lr=args.lr_geo, num_points=2000)

    # 3) Optionally train the refinement UNet.
    refiner = None
    if args.do_refiner:
        refiner = BasicUNetRefiner(in_channels=3, out_channels=1).to(device)
        ct_files = sorted(glob.glob(os.path.join(args.data_folder_ct, "topcow_ct_***.nii.gz")))
        if len(ct_files) != len(seg_files) or len(ct_files) == 0:
            print("CT files are missing or count does not match. Skipping refinement training.")
        else:
            train_refinement_unet(refiner, ct_paths=ct_files, seg_paths=seg_files, device=device, epochs=args.epochs_refiner, lr=args.lr_refiner)

    # 4) Run inference on a segmentation (if infer_folder is provided, use that; otherwise use the ground truth/synthetic file).
    if args.infer_folder is not None:
        infer_seg_files = sorted(glob.glob(os.path.join(args.infer_folder, "topcow_ct_***.nii.gz")))
        if len(infer_seg_files) == 0:
            print("No segmentation files found in the inference folder. Exiting.")
            return
        new_seg = infer_seg_files[0]
        print(f"Using imperfect segmentation from inference folder: {new_seg}")
    else:
        new_seg = seg_file
        print("Using segmentation file for inference (no infer_folder provided).")
    
    new_ct = None
    if args.data_folder_ct:
        ct_files = sorted(glob.glob(os.path.join(args.data_folder_ct, "topcow_ct_***.nii.gz")))
        if len(ct_files) > 0:
            new_ct = ct_files[0]
    inference_dummy(geo_model, refiner, new_seg, new_ct, recon_resolution=mesh_res, device=device)

if __name__ == "__main__":
    main()
