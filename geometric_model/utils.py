import nibabel as nib
import torch
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_nifti_as_tensor(file_path):
    nii = nib.load(file_path)
    arr = nii.get_fdata(dtype=np.float32)
    # Convert to binary mask (vessel=1, background=0)
    arr = (arr > 0).astype(np.float32)
    arr_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return arr_t

def compute_target_sdf(seg_mask, clip_threshold=10):
    """
    Compute a signed distance field (SDF) from a binary segmentation mask.
    The SDF is computed as:
         sdf(x) = distance(x, vessel) - distance(x, background)
    so that points inside the vessel have negative values and those
    outside have positive values.
    
    Args:
      seg_mask: torch.Tensor of shape [B, 1, H, W, D]
      clip_threshold: maximum absolute value for SDF clipping.
      
    Returns:
      torch.Tensor of shape [B, 1, H, W, D] containing the computed SDF.
    """
    seg_mask_np = seg_mask.cpu().numpy()  # shape: [B,1,H,W,D]
    bin_mask = seg_mask_np > 0
    sdf_np = np.zeros_like(bin_mask, dtype=np.float32)
    for b in range(bin_mask.shape[0]):
        vol = bin_mask[b, 0]
        dt_out = ndi.distance_transform_edt(~vol)    # distance to background
        dt_in = ndi.distance_transform_edt(vol)      # distance to vessel
        # sdf = np.clip(dt_out - dt_in, -clip_threshold, clip_threshold)
        normalized_sdf = np.clip(dt_out - dt_in, -clip_threshold, clip_threshold) / clip_threshold
        sdf_np[b, 0] = normalized_sdf
        # neg_count = np.sum(normalized_sdf < 0)
        # pos_count = np.sum(normalized_sdf >= 0)
        # total = normalized_sdf.size
        # print(f"Shape {b}: Negative voxels: {neg_count} ({neg_count/total:.2%}), Positive voxels: {pos_count} ({pos_count/total:.2%})")
    return torch.from_numpy(sdf_np).to(seg_mask.device)

def visualize_sdf_slice(decoder, latent_code, seg_mask, slice_idx=None, device="cuda", save_path=None):
    """
    Visualizes one slice of the SDF: ground truth vs. predicted.
    Args:
      decoder: The trained decoder model (expects coords in [-1,1]).
      latent_code: A tensor of shape [1, latent_size] for the shape.
      seg_mask: The ground truth segmentation mask as a torch.Tensor of shape [1,1,H,W,D].
      slice_idx: Index of the depth slice to visualize. If None, selects the slice with the most voxels.
      device: "cuda" or "cpu".
      save_path: If provided, saves the plot to a file.
    """
    # Compute the ground truth SDF from the segmentation
    with torch.no_grad():
        sdf_gt = compute_target_sdf(seg_mask).squeeze(0).squeeze(0)  # [H,W,D]
    sdf_gt_np = sdf_gt.cpu().numpy()
    H, W, D = sdf_gt_np.shape

    # If slice_idx is None, select the slice with the most voxels in the segmentation
    seg_np = seg_mask.squeeze(0).squeeze(0).cpu().numpy()  # shape [H, W, D]
    if slice_idx is None:
        z_sums = np.sum(seg_np, axis=(0, 1))  # sum over x,y for each slice in z
        slice_idx = int(np.argmax(z_sums))
        print(f"Selected slice_idx {slice_idx} with maximum voxels (count: {z_sums[slice_idx]})")
    else:
        print(f"Using provided slice_idx: {slice_idx}")
    
    # Ground truth slice for visualization
    target_slice = sdf_gt_np[:, :, slice_idx]

    # -------------------------------------------------------------------------
    # 1) Create grid in the ORIGINAL dimension space: 0..H-1, 0..W-1.
    # -------------------------------------------------------------------------
    xs = np.arange(H)
    ys = np.arange(W)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')  # each of shape (H,W)
    coords_2d = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # shape [H*W, 2]
    z_col = np.full((coords_2d.shape[0], 1), slice_idx, dtype=np.float32)

    # Combine to form 3D coordinates (voxel indices)
    coords_3d = np.concatenate([coords_2d, z_col], axis=-1)  # shape [H*W, 3]

    # -------------------------------------------------------------------------
    # 2) Re-normalize these coordinates into [-1,1] for each axis:
    # -------------------------------------------------------------------------
    coords_3d_t = torch.from_numpy(coords_3d).float().to(device)
    coords_3d_norm = coords_3d_t.clone()
    coords_3d_norm[:, 0] = coords_3d_norm[:, 0] / max(H - 1, 1) * 2.0 - 1.0
    coords_3d_norm[:, 1] = coords_3d_norm[:, 1] / max(W - 1, 1) * 2.0 - 1.0
    coords_3d_norm[:, 2] = coords_3d_norm[:, 2] / max(D - 1, 1) * 2.0 - 1.0

    # -------------------------------------------------------------------------
    # 3) Feed normalized coordinates to the decoder.
    # -------------------------------------------------------------------------
    latent_expanded = latent_code.to(device).expand(coords_3d_norm.shape[0], -1)
    inputs = torch.cat([latent_expanded, coords_3d_norm], dim=1)

    with torch.no_grad():
        pred_sdf = decoder(inputs).squeeze(1).cpu().numpy().reshape(H, W)

    # -------------------------------------------------------------------------
    # 4) Plot side-by-side: ground truth slice vs. reconstructed slice.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    im0 = ax[0].imshow(target_slice, cmap="jet", origin="upper")
    ax[0].set_title("Ground Truth SDF (Slice)")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(pred_sdf, cmap="jet", origin="upper")
    ax[1].set_title("Reconstructed SDF (Slice)")
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"SDF slice visualization saved to {save_path}")
    else:
        plt.show()

def plot_loss_curve(iteration_loss_log, epoch_loss_log, window=10, save_path="training_loss_curve.png"):
    """
    Plots:
      1) Per-iteration loss (fine-grained) in a faint color.
      2) Per-epoch average loss in bold.
      3) Rolling average of per-epoch loss over 'window' epochs, for smoothing.
    """
    import torch

    # Convert any torch.Tensor entries into Python floats
    iteration_loss_log = [
        x.item() if isinstance(x, torch.Tensor) else float(x)
        for x in iteration_loss_log
    ]
    epoch_loss_log = [
        x.item() if isinstance(x, torch.Tensor) else float(x)
        for x in epoch_loss_log
    ]

    def rolling_avg(values, window_size):
        # Simple 1D rolling average using convolution
        return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot epoch-level average (of all shapes)
    epoch_count = len(epoch_loss_log)
    epoch_indices = np.linspace(0, len(iteration_loss_log) - 1, epoch_count)
    ax.plot(epoch_indices, epoch_loss_log, label='Per-epoch avg loss',
            color='red', linewidth=2, alpha=0.6)

    # Rolling average of epoch loss
    if epoch_count >= window:
        rolled = rolling_avg(epoch_loss_log, window)
        rolled_x = np.linspace(0, len(iteration_loss_log) - 1, len(rolled))
        ax.plot(rolled_x, rolled, label=f'{window}-epoch rolling avg',
                color='green', linewidth=2)
    else:
        print(f"Not enough epochs ({epoch_count}) to compute a {window}-epoch rolling average.")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Time")
    ax.legend()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Loss curve saved to {save_path}")
    
def test_latent_sensitivity(decoder, latent_code, coords):
    with torch.no_grad():
        input_orig = torch.cat([latent_code.expand(coords.shape[0], -1), coords], dim=1)
        original_output = decoder(input_orig)
        noise = torch.randn_like(latent_code) * 0.01
        perturbed_code = latent_code + noise
        input_perturbed = torch.cat([perturbed_code.expand(coords.shape[0], -1), coords], dim=1)
        perturbed_output = decoder(input_perturbed)
        diff = torch.mean(torch.abs(original_output - perturbed_output)).item()
    print("Average difference with latent perturbation:", diff)

def plot_latent_distribution(lat_vecs, save_path="latent_distribution.png"):
    latent_np = lat_vecs.weight.data.cpu().numpy()
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_np)
    plt.figure(figsize=(6, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
    plt.title("Latent Distribution (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved latent distribution plot to", save_path)