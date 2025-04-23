import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from geometric_model.geometric_model import compute_target_sdf

def load_nifti_as_tensor(file_path):
    """
    Loads a .nii.gz file using nibabel and returns a torch.FloatTensor of shape [1, 1, H, W, D].
    """
    nii = nib.load(file_path)
    arr = nii.get_fdata(dtype=np.float32)
    arr_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return arr_t

def plot_loss_curves(epochs, recon_losses, kl_losses, total_losses, save_path="loss_curves.png"):
    plt.figure()
    plt.plot(epochs, recon_losses, label="Reconstruction Loss")
    plt.plot(epochs, kl_losses, label="KL Loss")
    plt.plot(epochs, total_losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
def visualize_sdf_predictions(seg_mask, sdf_pred, coords, save_path="sdf_comparison.png"):
    """
    Visualizes a central slice (along the depth axis) of the target SDF computed from the GT segmentation
    and overlays predicted SDF values (for points near the central slice).
    """
    target_sdf = compute_target_sdf(seg_mask)
    target_sdf_np = target_sdf.squeeze().detach().cpu().numpy()  # shape [H, W, D]
    H, W, D = target_sdf_np.shape
    mid = D // 2
    target_slice = target_sdf_np[:, :, mid]
    
    coords_np = coords.squeeze(0).detach().cpu().numpy()  # [num_points, 3]
    sdf_pred_np = sdf_pred.squeeze(0).detach().cpu().numpy()  # [num_points]
    close_idx = np.where(np.abs(coords_np[:, 2] - mid) < 3)[0]
    coords_close = coords_np[close_idx]
    sdf_pred_close = sdf_pred_np[close_idx]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(target_slice, cmap="jet")
    plt.title("Target SDF (central slice)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(target_slice, cmap="jet")
    plt.scatter(coords_close[:, 0], coords_close[:, 1], c=sdf_pred_close, cmap="coolwarm", edgecolor="k")
    plt.title("Predicted SDF Overlay")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
    
def plot_latent_distribution(latent_set, save_path="latent_distribution.png"):
    latent_np = latent_set.squeeze(0).detach().cpu().numpy()  # shape [M, latent_dim]
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_np)
    plt.figure()
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Set Distribution (PCA)")
    plt.savefig(save_path)
    plt.close()

#############################################
# Visualization: Save Full SDF Volume as NIfTI
#############################################
def save_full_sdf_nifti(sdf_volume, save_path="full_sdf_volume.nii.gz"):
    """
    Saves a full 3D SDF volume as a NIfTI file.
    
    Args:
      sdf_volume: A torch.Tensor or numpy array of shape [H, W, D] representing the SDF volume.
      save_path: The file path where the NIfTI file will be saved.
    """
    # Convert to numpy if it's a tensor.
    if torch.is_tensor(sdf_volume):
        sdf_volume = sdf_volume.detach().cpu().numpy()
    
    # Use an identity affine for now (adjust if your volumes have specific spatial info)
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(sdf_volume, affine)
    nib.save(nii_img, save_path)
    print(f"Full SDF volume saved as {save_path}")
    
def save_full_sdf_nifti_wrapper(geo_model, seg_mask, latent_set, output_resolution, save_path):
    full_sdf = reconstruct_full_sdf_volume(geo_model, seg_mask, latent_set, output_resolution=output_resolution)
    save_full_sdf_nifti(full_sdf, save_path=save_path)

#############################################
# Reconstruct Full SDF Volume from Decoder
#############################################
def reconstruct_full_sdf_volume(geo_model, seg_mask, latent_set, output_resolution=(128,128,128)):
    """
    Generates a dense grid of coordinates over seg_mask's volume, uses the decoder with latent_set
    to predict SDF values, and returns a full SDF volume of shape output_resolution.
    """
    B, C, H, W, D = seg_mask.shape
    H_r, W_r, D_r = output_resolution
    xs = torch.linspace(0, H-1, H_r)
    ys = torch.linspace(0, W-1, W_r)
    zs = torch.linspace(0, D-1, D_r)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)  # [H_r, W_r, D_r, 3]
    grid = grid.view(1, -1, 3).to(seg_mask.device)  # [1, num_points, 3]
    sdf_pred, _ = geo_model.decoder(latent_set, grid)
    sdf_volume = sdf_pred.view(H_r, W_r, D_r).detach().cpu()
    return sdf_volume