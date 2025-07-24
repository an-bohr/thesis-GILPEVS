# import os
# import torch
# from torch.utils.data import Dataset
# import nibabel as nib
# import numpy as np
# import skimage
# import trimesh

# class TopCowDataset(Dataset):
#     """
#     Dataset of Circle-of-Willis segmentation masks in NIfTI format.
#     For each volume:
#       1) extract mesh (marching cubes at iso=0.5)
#       2) sample balanced points+SDF (surface vs. noisy off-surface)
#       3) optionally apply AxisScaling transform
#     Returns (points, sdf_labels, surface_pc, dummy_label)
#     """
#     def __init__(self,
#                  nifti_dir,
#                  split='train',
#                  transform=None,
#                  num_samples=2048,
#                  clip_threshold=10.0,
#                  noise_std=0.3):
#         super().__init__()
#         self.transform       = transform
#         self.num_samples     = num_samples
#         self.clip_threshold  = clip_threshold
#         self.noise_std       = noise_std

#         # assume structure: nifti_dir/train/*.nii.gz  and  nifti_dir/val/*.nii.gz
#         self.paths = sorted([
#             os.path.join(nifti_dir, split, fname)
#             for fname in os.listdir(os.path.join(nifti_dir, split))
#             if fname.endswith('.nii.gz')
#         ])
        
#         self.cache = []
#         for pth in self.paths:
#             # 1) load binary mask
#             vol = nib.load(pth).get_fdata().astype(np.float32)
#             # 2) marching cubes to get mesh at iso=0.5
#             verts, faces, _, _ = skimage.measure.marching_cubes(vol, 0.5)
#             mesh = trimesh.Trimesh(verts, faces, process=True)

#             # 3) sample SDF + coords + normals
#             coords, sdf, normals = sample_sdf_balanced_mesh(
#                 vol_shape=vol.shape,
#                 mesh=mesh,
#                 num_points=self.num_samples * 2,   # half surface, half noisy
#                 clip_threshold=self.clip_threshold,
#                 noise_std=self.noise_std)
#             # surf_pts = mesh.sample(self.num_samples)
#             # 4) sample exactly N surface points + normals via same sample_surface call
#             #    NOTE: sample_surface returns pts_surf_2, face_idx_2 as new draws; but we want a
#             #    consistent surface point cloud for supervision. We’ll use mesh.sample(N) for pts
#             #    and then find their normals by nearest‐face lookup. This is slightly different
#             #    from the training‐time SDF sample, but acceptable. Alternatively, you can re‐call
#             #    trimesh.sample.sample_surface(mesh, N) to get pts + face_idx.
#             #
#             surf_pts, face_idx = trimesh.sample.sample_surface(mesh, self.num_samples)
#             normals = mesh.face_normals[face_idx].astype(np.float32)
        
#             # 4) normalize surface pts into [-1,1]
#             H, W, D = vol.shape
#             surf_pts = surf_pts.astype(np.float32)
#             surf_pts[:, 0] = (surf_pts[:,0]/(H-1))*2 - 1
#             surf_pts[:, 1] = (surf_pts[:,1]/(W-1))*2 - 1
#             surf_pts[:, 2] = (surf_pts[:,2]/(D-1))*2 - 1
            
#             self.cache.append((coords.astype(np.float32), sdf.astype(np.float32),
#                       surf_pts.astype(np.float32), normals.astype(np.float32)))

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         pth = self.paths[idx]
#         fname = os.path.basename(pth)
        
#         # To only load the mesh once
#         coords, sdf, surface, normals = self.cache[idx]
        
#         if self.transform:
#           coords, surface, normals = self.transform(coords, surface, normals)
        
#         coords = torch.from_numpy(coords).float()      # [2*num_samples, 3]
#         sdf = torch.from_numpy(sdf).float()      # [2*num_samples]
#         surface = torch.from_numpy(surface).float()        # [num_samples, 3]
#         normals = torch.from_numpy(normals).float()        # [num_samples, 3]



#         return coords, sdf, surface, normals, fname

# def sample_sdf_balanced_mesh(vol_shape, mesh, num_points, clip_threshold=10.0, noise_std=0.3):
#     """
#     Sample SDF values using a mesh representation:
#       - Half of the points exactly on the surface (SDF=0).
#       - Half near the surface by adding Gaussian noise in normalized space.
#     Signed distances are computed via trimesh proximity queries.

#     Args:
#       vol_shape: tuple (H, W, D) of cropped volume.
#       mesh: trimesh.Trimesh mesh of the surface.
#       num_points: Total points to sample.
#       clip_threshold: max abs-distance (in voxels) before normalization.
#       noise_std: standard deviation of noise in normalized coords.

#     Returns:
#       coords_norm: (num_points, 3) in [-1,1]
#       sdf_samples: (num_points,) in [-1,1]
#     """
#     H, W, D = vol_shape
#     half = num_points // 2

#     # 1) Sample surface points uniformly
#     pts_surf, face_idx = trimesh.sample.sample_surface(mesh, half)
#     # Surface SDF values are zero
#     sdf_surf = np.zeros(half, dtype=np.float32)

#     # 2) Compute normals at those sample points
#     normals_surface = mesh.face_normals[face_idx].astype(np.float32)

#     # 3) Convert surface points to normalized coords
#     pts_surf_norm = pts_surf.astype(np.float32)
#     pts_surf_norm[:, 0] = (pts_surf_norm[:, 0] / (H - 1)) * 2 - 1
#     pts_surf_norm[:, 1] = (pts_surf_norm[:, 1] / (W - 1)) * 2 - 1
#     pts_surf_norm[:, 2] = (pts_surf_norm[:, 2] / (D - 1)) * 2 - 1

#     # 4) Add noise in normalized space and clip
#     pts_noisy_norm = pts_surf_norm + np.random.normal(scale=noise_std, size=pts_surf_norm.shape)
#     pts_noisy_norm = np.clip(pts_noisy_norm, -1.0, 1.0)

#     # 5) Map noisy points back to voxel coords
#     pts_noisy_vox = (pts_noisy_norm + 1) / 2 * np.array([H - 1, W - 1, D - 1])

#     # 6) Compute signed distances using mesh proximity
#     pq = trimesh.proximity.ProximityQuery(mesh)
#     sdf_noisy_vox = pq.signed_distance(pts_noisy_vox)

#     # 7) Clip and normalize SDF values
#     sdf_noisy = np.clip(sdf_noisy_vox, -clip_threshold, clip_threshold) / clip_threshold

#     # 8) Stack results
#     coords_norm = np.vstack([pts_surf_norm, pts_noisy_norm])
#     sdf_samples = np.concatenate([sdf_surf, sdf_noisy])
    
#     coords_norm = coords_norm.astype(np.float32)
#     sdf_samples = sdf_samples.astype(np.float32)

#     return coords_norm, sdf_samples, normals_surface

import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import skimage.measure
import trimesh

class TopCowDataset(Dataset):
    """
    Dataset of Circle-of-Willis segmentation masks in NIfTI format.
    For each volume:
      1) extract mesh (marching cubes at iso=0.5)
      2) sample balanced points+SDF (surface vs. noisy off-surface)
      3) optionally apply AxisScaling transform
    Returns (points, sdf_labels, surface_pc, normals, fname)
    """
    def __init__(self,
                 nifti_dir,
                 transform=None,
                 num_samples=2048,
                 clip_threshold=10.0,
                 noise_std=0.3):
        super().__init__()
        self.transform      = transform
        self.num_samples    = num_samples
        self.clip_threshold = clip_threshold
        self.noise_std      = noise_std

        # assume all .nii.gz files in nifti_dir
        self.paths = sorted([
            os.path.join(nifti_dir, fname)
            for fname in os.listdir(nifti_dir)
            if fname.endswith('.nii.gz')
        ])

        # preload mesh + SDF samples for each volume
        self.cache = []
        for pth in self.paths:
            # load binary mask
            vol = nib.load(pth).get_fdata().astype(np.float32)
            # marching cubes to get mesh at iso=0.5
            verts, faces, _, _ = skimage.measure.marching_cubes(vol, 0.5)
            mesh = trimesh.Trimesh(verts, faces, process=True)

            # sample SDF + coords
            coords, sdf, normals = sample_sdf_balanced_mesh(
                vol_shape=vol.shape,
                mesh=mesh,
                num_points=self.num_samples * 2,
                clip_threshold=self.clip_threshold,
                noise_std=self.noise_std)

            # sample surface point cloud + normals
            surf_pts, face_idx = trimesh.sample.sample_surface(mesh, self.num_samples)
            normals_surf = mesh.face_normals[face_idx].astype(np.float32)

            # normalize surf_pts to [-1,1]
            H, W, D = vol.shape
            surf_pts[:, 0] = (surf_pts[:,0] / (H-1)) * 2 - 1
            surf_pts[:, 1] = (surf_pts[:,1] / (W-1)) * 2 - 1
            surf_pts[:, 2] = (surf_pts[:,2] / (D-1)) * 2 - 1

            self.cache.append((coords.astype(np.float32),
                               sdf.astype(np.float32),
                               surf_pts.astype(np.float32),
                               normals_surf))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        coords, sdf, surface, normals = self.cache[idx]
        fname = os.path.basename(self.paths[idx])

        if self.transform:
            coords, surface, normals = self.transform(coords, surface, normals)

        # to Tensor
        coords  = torch.from_numpy(coords).float()
        sdf     = torch.from_numpy(sdf).float()
        surface = torch.from_numpy(surface).float()
        normals = torch.from_numpy(normals).float()

        return coords, sdf, surface, normals, fname


def sample_sdf_balanced_mesh(vol_shape, mesh, num_points,
                             clip_threshold=10.0, noise_std=0.3):
    """
    Sample SDF values using a mesh representation:
      - Half points on surface (SDF=0)
      - Half near surface (noisy)
    Returns normalized coords & SDF in [-1,1].
    """
    H, W, D = vol_shape
    half = num_points // 2

    # 1) surface samples
    pts_surf, face_idx = trimesh.sample.sample_surface(mesh, half)
    sdf_surf = np.zeros(half, dtype=np.float32)
    normals_surface = mesh.face_normals[face_idx].astype(np.float32)

    # normalize to [-1,1]
    pts_surf_norm = pts_surf.copy().astype(np.float32)
    pts_surf_norm[:,0] = (pts_surf_norm[:,0] / (H-1)) * 2 - 1
    pts_surf_norm[:,1] = (pts_surf_norm[:,1] / (W-1)) * 2 - 1
    pts_surf_norm[:,2] = (pts_surf_norm[:,2] / (D-1)) * 2 - 1

    # 2) noisy off-surface
    pts_noisy_norm = pts_surf_norm + np.random.normal(scale=noise_std, size=pts_surf_norm.shape)
    pts_noisy_norm = np.clip(pts_noisy_norm, -1.0, 1.0)
    pts_noisy_vox = (pts_noisy_norm + 1) / 2 * np.array([H-1, W-1, D-1])

    pq = trimesh.proximity.ProximityQuery(mesh)
    sdf_noisy_vox = pq.signed_distance(pts_noisy_vox)
    sdf_noisy = np.clip(sdf_noisy_vox, -clip_threshold, clip_threshold) / clip_threshold

    # stack
    coords_norm = np.vstack([pts_surf_norm, pts_noisy_norm]).astype(np.float32)
    sdf_samples = np.concatenate([sdf_surf, sdf_noisy]).astype(np.float32)

    return coords_norm, sdf_samples, normals_surface
