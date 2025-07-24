import os
import glob
import argparse
import json
import numpy as np
import torch
import nibabel as nib
import scipy.ndimage as ndi
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        normalized_sdf = np.clip(dt_out - dt_in, -clip_threshold, clip_threshold) / clip_threshold
        sdf_np[b, 0] = normalized_sdf
    return torch.from_numpy(sdf_np).to(seg_mask.device)

def find_bounding_box(volume, padding=5, debug_arg=False):
    """
    Computes the tight bounding box of nonzero voxels (vessel)
    and applies extra padding.
    """
    coords = np.argwhere(volume > 0)
    if coords.size == 0:
        return (0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2])
    min_x, min_y, min_z = coords.min(axis=0)
    max_x, max_y, max_z = coords.max(axis=0)
    min_x = max(min_x - padding, 0)
    min_y = max(min_y - padding, 0)
    min_z = max(min_z - padding, 0)
    max_x = min(max_x + padding, volume.shape[0] - 1)
    max_y = min(max_y + padding, volume.shape[1] - 1)
    max_z = min(max_z + padding, volume.shape[2] - 1)
    if debug_arg:
        print(f"Bounding box: x=({min_x}, {max_x}), y=({min_y}, {max_y}), z=({min_z}, {max_z})")
    return (min_x, max_x, min_y, max_y, min_z, max_z)

def crop_volume(volume, bbox, debug_arg=False):
    """
    Crops the volume using the provided bounding box.
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    cropped = volume[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    if debug_arg:
        print(f"Cropped volume shape: {cropped.shape}")
    return cropped

def sample_sdf_balanced(sdf, num_points, noise_std=0.3, debug_arg=False):
    """
    Samples points half exactly on the vessel surface (smallest nonzero |SDF|) 
    and half near the surface by applying Gaussian noise to those surface points.

    Args:
      sdf: torch.Tensor of shape [1, 1, H, W, D], normalized in [-1,1]
      num_points: Total number of points to sample.
      noise_std: Standard deviation for Gaussian noise (in normalized coords).

    Returns:
      coords_norm: numpy array of shape [num_points, 3] with coordinates in [-1,1].
      sdf_samples: numpy array of shape [num_points] with corresponding SDF values.
    """
    # Convert SDF to numpy
    sdf_np = sdf.squeeze().cpu().numpy()  # shape [H, W, D]
    H, W, D = sdf_np.shape

    # 1) Find the smallest nonzero absolute SDF value => surface level
    abs_vals = np.abs(sdf_np)
    nonzero = abs_vals[abs_vals > 1e-6]
    if nonzero.size == 0:
        raise ValueError("No nonzero SDF values found.")
    surface_val = nonzero.min()

    # 2) Gather surface voxel indices
    surface_mask = np.isclose(abs_vals, surface_val, atol=1e-6)
    surface_indices = np.argwhere(surface_mask)
    if surface_indices.shape[0] == 0:
        raise ValueError("No surface points found.")

    half = num_points // 2

    # 3a) Sample half exactly at surface
    idx_surf = np.random.choice(surface_indices.shape[0], size=half, 
                                 replace=surface_indices.shape[0] < half)
    pts_surf = surface_indices[idx_surf]  # integer voxel coords

    # 3b) Convert those to normalized [-1,1] coordinates
    pts_surf_norm = pts_surf.astype(np.float32)
    pts_surf_norm[:, 0] = (pts_surf_norm[:, 0] / (H - 1)) * 2 - 1
    pts_surf_norm[:, 1] = (pts_surf_norm[:, 1] / (W - 1)) * 2 - 1
    pts_surf_norm[:, 2] = (pts_surf_norm[:, 2] / (D - 1)) * 2 - 1

    # 4) Create noisy points around the surface samples
    pts_noisy_norm = pts_surf_norm + np.random.normal(scale=noise_std, size=pts_surf_norm.shape)
    pts_noisy_norm = np.clip(pts_noisy_norm, -1.0, 1.0)

    # 5) Compute ground-truth SDF values:
    #   - surface points: directly from sdf_np
    sdf_surf = sdf_np[pts_surf[:,0], pts_surf[:,1], pts_surf[:,2]]
    #   - noisy points: trilinear interpolate using SciPy
    #     map_coordinates expects coords per axis
    #     convert back to voxel space
    coords_voxel = (pts_noisy_norm + 1) / 2 * np.array([H-1, W-1, D-1])
    sdf_noisy = ndi.map_coordinates(
        sdf_np, 
        [coords_voxel[:,0], coords_voxel[:,1], coords_voxel[:,2]],
        order=1, mode='nearest'
    )

    # 6) Stack results
    coords_norm = np.vstack([pts_surf_norm, pts_noisy_norm])
    sdf_samples = np.concatenate([sdf_surf, sdf_noisy])

    # Optional: debug histogram
    if debug_arg:
        plt.figure()
        plt.hist(sdf_samples, bins=50, edgecolor='black')
        plt.title("Surface + Noisy SDF Histogram")
        plt.xlabel("SDF value")
        plt.ylabel("Frequency")
        plt.savefig("surface_noisy_histogram.png", dpi=150)
        plt.close()
        print("Histogram saved to surface_noisy_histogram.png")

    return coords_norm, sdf_samples

def process_segmentation(nifti_file, output_folder, num_points, clip_threshold, debug_arg):
    """
    Process one segmentation file:
      1. Loads the segmentation.
      2. Crops the volume to the vessel region.
      3. Computes the SDF.
      4. Samples a balanced set of points from a narrow band around the boundary.
      5. Saves the resulting samples in an npz file.
    """
    nii = nib.load(nifti_file)
    arr = nii.get_fdata(dtype=np.float32)
    # Convert to binary: vessel=1, background=0
    arr = (arr > 0).astype(np.float32)
    
    # Crop to vessel bounding box
    bbox = find_bounding_box(arr, padding=10, debug_arg=debug_arg)
    arr_cropped = crop_volume(arr, bbox, debug_arg=debug_arg)
    
    seg_tensor = torch.from_numpy(arr_cropped).unsqueeze(0).unsqueeze(0)  # shape [1,1,Hc,Wc,Dc]
    sdf_tensor = compute_target_sdf(seg_tensor, clip_threshold=clip_threshold)
    
    # Sample balanced narrow band points
    coords, sdf_samples = sample_sdf_balanced(sdf_tensor, num_points=num_points, noise_std=0.3, debug_arg=debug_arg)
    
    base = os.path.basename(nifti_file)
    name, ext = os.path.splitext(base)
    if ext == '.gz':
        name, _ = os.path.splitext(name)
    out_file = os.path.join(output_folder, name + ".npz")
    np.savez_compressed(out_file, pos=coords, neg=sdf_samples)
    return name

def main():
    parser = argparse.ArgumentParser(description="Prepare balanced SDF samples from segmentation masks.")
    parser.add_argument("--input_folder", required=True, help="Folder containing ground truth segmentation NIfTI files.")
    parser.add_argument("--output_folder", required=True, help="Folder to save the npz SDF sample files.")
    parser.add_argument("--num_points", type=int, default=25000, help="Number of SDF sample points per segmentation.")
    parser.add_argument("--clip_threshold", type=float, default=10.0, help="Clipping threshold for SDF values.")
    parser.add_argument("--latent_dim", type=int, default=256, help="Dimension of latent codes.")
    parser.add_argument("--init_latents", action="store_true", help="If set, initialize latent codes and save them.")
    parser.add_argument("--latent_output", default="latent_codes.npz", help="Output file for latent codes if --init_latents is set.")
    parser.add_argument("--mapping_file", default="mapping.json", help="Output mapping file (segmentation identifier to latent index).")
    parser.add_argument("--debug", default=False, action="store_true", help="If set, debug mode will be activated.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    
    nifti_files = glob.glob(os.path.join(args.input_folder, "*.nii.gz"))
    mapping = {}
    latent_list = []
    
    n = len(nifti_files)
    pbar = tqdm(nifti_files, desc="Processing segmentation masks")
    for idx, nifti_file in enumerate(pbar, start=1):
        name = process_segmentation(nifti_file,args.output_folder,num_points=args.num_points,
                                    clip_threshold=args.clip_threshold,debug_arg=args.debug)
        mapping[name] = idx
        if args.init_latents:
            latent = np.random.normal(
                0, 1/np.sqrt(args.latent_dim),
                size=(args.latent_dim,)
            )
            latent_list.append(latent)

        if idx % 10 == 0:
            tqdm.write(f"[{idx}/{n}] processed segmentation mask: {nifti_file}")
    
    mapping_path = os.path.join(args.output_folder, args.mapping_file)
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Mapping file saved to {mapping_path}")

    if args.init_latents:
        latent_array = np.stack(latent_list, axis=0)
        latent_path = os.path.join(args.output_folder, args.latent_output)
        np.savez_compressed(latent_path, latent_codes=latent_array)
        print(f"Initial latent codes saved to {latent_path}")

if __name__ == "__main__":
    main()
