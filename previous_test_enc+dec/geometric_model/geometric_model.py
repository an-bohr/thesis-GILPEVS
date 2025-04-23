import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import measure

#############################################
# Helper: Compute Target SDF from a segmentation mask
#############################################
def compute_target_sdf(seg_mask):
    """
    Compute a signed distance field (SDF) from a multi-label segmentation mask.
    
    Preprocessing:
      - Converts a multi-label mask into a binary mask (vessel if label > 0, background if 0).
    
    The SDF is defined as:
         sdf(x) = distance(x, foreground) - distance(x, background)
    yielding negative values inside vessel regions and positive outside.
    
    Args:
      seg_mask: torch.Tensor of shape [B, 1, H, W, D]
      
    Returns:
      torch.Tensor of shape [B, 1, H, W, D] containing the computed SDF.
    """
    seg_mask_np = seg_mask.cpu().numpy()  # [B,1,H,W,D]
    bin_mask = seg_mask_np > 0
    
    sdf_np = np.zeros_like(bin_mask, dtype=np.float32)
    for b in range(bin_mask.shape[0]):
        vol = bin_mask[b, 0]
        dt_out = ndi.distance_transform_edt(~vol)
        dt_in = ndi.distance_transform_edt(vol)
        # sdf = dt_in - dt_out
        sdf = np.clip(dt_in - dt_out, -50, 50)          # testing to see if calmping my SDF will help
        sdf_np[b, 0] = sdf
    return torch.from_numpy(sdf_np).to(seg_mask.device)
  
def sample_target_sdf(target_sdf, coords):
    """
    Samples the full target SDF at the given coordinates using trilinear interpolation.
    
    Args:
      target_sdf: torch.Tensor of shape [B, 1, H, W, D] (full target SDF volume).
      coords: torch.Tensor of shape [B, N, 3] in pixel coordinates (range 0 to H-1, etc.)
    
    Returns:
      torch.Tensor of shape [B, N] with the interpolated SDF values.
    """
    B, _, H, W, D = target_sdf.shape
    # Normalize coordinates to the range [-1, 1] as required by grid_sample.
    norm_coords = coords.clone()
    norm_coords[..., 0] = 2.0 * coords[..., 0] / (H - 1) - 1.0
    norm_coords[..., 1] = 2.0 * coords[..., 1] / (W - 1) - 1.0
    norm_coords[..., 2] = 2.0 * coords[..., 2] / (D - 1) - 1.0
    # grid_sample expects shape [B, N, 1, 1, 3] for 3D volumes.
    norm_coords = norm_coords.view(B, -1, 1, 1, 3)
    sampled = F.grid_sample(target_sdf, norm_coords, mode='bilinear', align_corners=True)
    return sampled.view(B, -1)

#############################################
# Sampling Strategy: 50% Uniform, 50% Near Boundary
#############################################
def sample_coords(seg_mask, num_points, device='cuda', boundary_threshold=5):
    """
    Samples coordinates in a 3D volume in two parts:
      - 50% uniformly at random over the entire volume.
      - 50% from voxels near the vessel boundary (i.e. where |SDF| < boundary_threshold).
      
    Args:
      seg_mask: torch.Tensor of shape [1, 1, H, W, D] (binary segmentation, 0/1).
      num_points: total number of points to sample.
      device: device for returned tensor.
      boundary_threshold: threshold for absolute SDF to consider a voxel “near” the boundary.
      
    Returns:
      coords: torch.Tensor of shape [1, num_points, 3] in pixel coordinates.
    """
    B, _, H, W, D = seg_mask.shape
    num_uniform = num_points // 2
    num_boundary = num_points - num_uniform
    
    # Uniform sampling
    uniform_coords = torch.rand(1, num_uniform, 3, device=device)
    uniform_coords[..., 0] *= H
    uniform_coords[..., 1] *= W
    uniform_coords[..., 2] *= D
    
    # For boundary sampling, compute the full SDF (using your compute_target_sdf)
    sdf = compute_target_sdf(seg_mask)  # shape [1, 1, H, W, D]
    sdf_np = sdf.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H,W,D]
    # Identify indices where |sdf| < boundary_threshold.
    boundary_indices = np.argwhere(np.abs(sdf_np) < boundary_threshold)
    if len(boundary_indices) == 0:
        # If no boundary points, fallback to all foreground points.
        seg_np = seg_mask.squeeze(0).squeeze(0).detach().cpu().numpy()
        boundary_indices = np.argwhere(seg_np > 0)
    # Randomly select num_boundary indices
    selected_indices = boundary_indices[np.random.choice(len(boundary_indices), size=num_boundary, replace=True)]
    # Convert to torch tensor (and float)
    boundary_coords = torch.tensor(selected_indices, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Combine the two sets of coordinates
    coords = torch.cat([uniform_coords, boundary_coords], dim=1)
    return coords


#############################################
# Encoder that produces a set of latent vectors via cross-attention
#############################################
class LatentSetEncoder(nn.Module):
    def __init__(self, latent_dim=128, M=256, input_channels=1, conv_channels=[16,32,64],
                 kernel_size=3, stride=2, padding=1, input_size=(284,327,243)):
        """
        Args:
          latent_dim: dimensionality of each latent vector.
          M: number of latent vectors (size of the latent set).
          input_size: expected spatial dimensions (H, W, D) of the segmentation volume.
        """
        super(LatentSetEncoder, self).__init__()
        self.input_size = input_size
        # CNN backbone to extract features from segmentation mask.
        layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        # Project the conv features to latent_dim.
        # We assume the conv output has shape [B, C, H', W', D']
        # We'll flatten the spatial dims.
        # To compute output dims, use a dummy tensor.
        dummy = torch.zeros(1, input_channels, *input_size)
        with torch.no_grad():
            dummy_out = self.conv(dummy)  # shape [1, C, H',W',D']
        self.conv_output_shape = dummy_out.shape  # (1, C, H',W',D')
        N = np.prod(dummy_out.shape[2:])  # number of patches
        C = dummy_out.shape[1]
        # Linear projection to latent_dim for each patch.
        self.feature_proj = nn.Linear(C, latent_dim)
        # Learnable query set: shape [M, latent_dim]
        self.latent_queries = nn.Parameter(torch.randn(M, latent_dim))
    
    def forward(self, seg_mask):
        """
        Args:
          seg_mask: tensor of shape [B, 1, H, W, D]
        Returns:
          latent_set: tensor of shape [B, M, latent_dim]
        """
        # Resize if necessary
        if seg_mask.shape[2:] != self.input_size:
            seg_mask = F.interpolate(seg_mask, size=self.input_size, mode='trilinear', align_corners=False)
        B = seg_mask.shape[0]
        features = self.conv(seg_mask)  # [B, C, H', W', D']
        B, C, *spatial = features.shape
        N = np.prod(spatial)
        features = features.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        # Project each patch feature to latent_dim.
        projected = self.feature_proj(features)  # [B, N, latent_dim]
        # Now use cross-attention: latent_queries are queries, projected features are keys and values.
        # Compute attention: for each batch, for each query vector in latent_queries,
        # compute dot-product with keys (projected) then apply softmax.
        # latent_queries: [M, latent_dim] -> expand to [B, M, latent_dim]
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, latent_dim]
        # keys: [B, N, latent_dim]
        # Compute scaled dot-product attention.
        d = queries.size(-1)
        attn_scores = torch.bmm(queries, projected.transpose(1,2)) / (d ** 0.5)  # [B, M, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M, N]
        # Values are the same as projected features.
        latent_set = torch.bmm(attn_weights, projected)  # [B, M, latent_dim]
        return latent_set

#############################################
# Decoder that aggregates the latent set to predict SDF for given coordinates
#############################################
class LatentSetDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        """
        Args:
          latent_dim: dimensionality of each latent vector (and aggregated feature).
          hidden_dim: hidden dimension for the decoder MLP.
        """
        super(LatentSetDecoder, self).__init__()
        # For each query coordinate, first compute an embedding.
        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim)  # map coordinate to same dimension as latents
        )
        # Linear layers to produce keys and values from latent set.
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        # Final MLP to predict [SDF, log_var] given aggregated latent and coordinate.
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # outputs: SDF, log variance
        )
    
    def forward(self, latent_set, coords):
        """
        Args:
          latent_set: tensor of shape [B, M, latent_dim]
          coords: tensor of shape [B, num_points, 3]
        Returns:
          sdf: tensor of shape [B, num_points]
          log_var: tensor of shape [B, num_points]
        """
        B, num_points, _ = coords.shape
        M = latent_set.size(1)
        # Embed coordinates: shape [B, num_points, latent_dim]
        coord_emb = self.coord_embed(coords)
        # Project latent set to keys and values.
        keys = self.key_proj(latent_set)    # [B, M, latent_dim]
        values = self.value_proj(latent_set)  # [B, M, latent_dim]
        # For each query (coordinate embedding), compute attention weights over the latent set.
        # Compute scaled dot-product between coord_emb and keys.
        d = coord_emb.size(-1)
        # For each query, compute dot product with each latent.
        # We need to compute: [B, num_points, latent_dim] x [B, latent_dim, M] => [B, num_points, M]
        attn_scores = torch.bmm(coord_emb, keys.transpose(1,2)) / (d ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_points, M]
        # Weighted sum of the values: [B, num_points, M] x [B, M, latent_dim] = [B, num_points, latent_dim]
        aggregated = torch.bmm(attn_weights, values)
        # Concatenate the original coordinate with aggregated latent.
        concat = torch.cat([coords, aggregated], dim=-1)  # [B, num_points, latent_dim+3]
        # Predict SDF and log variance.
        out = self.mlp(concat)  # [B, num_points, 2]
        sdf = out[..., 0]
        log_var = out[..., 1]
        return sdf, log_var

#############################################
# Combined Geometric Model using latent set
#############################################
class GeometricLatentSetModel(nn.Module):
    def __init__(self, latent_dim=128, M=512, input_size=(284,327,243), hidden_dim=256, decoder_type='mlp'):
        """
        Combines the encoder (which produces a set of latent vectors) and the decoder.
        """
        super(GeometricLatentSetModel, self).__init__()
        self.encoder = LatentSetEncoder(latent_dim=latent_dim, M=M, input_size=input_size)
        # In this version, we are not using any extra conditioning in the decoder.
        self.decoder = LatentSetDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
    
    def forward(self, seg_mask, coords):
        """
        Args:
          seg_mask: tensor [B, 1, H, W, D] (GT segmentation)
          coords: tensor [B, num_points, 3]
        Returns:
          sdf: tensor [B, num_points]
          log_var: tensor [B, num_points]
          latent_set: tensor [B, M, latent_dim]
        """
        latent_set = self.encoder(seg_mask)  # [B, M, latent_dim]
        sdf, log_var = self.decoder(latent_set, coords)
        return sdf, log_var, latent_set

    def forward_with_loss(self, seg_mask, coords):
        """
        Computes the negative log likelihood loss using the predicted SDF.
        Target SDF is computed from the segmentation (using compute_target_sdf).
        For simplicity, here we sample a dummy target (zeros) at the given coords.
        Returns:
          sdf_pred, reconstruction loss, KL loss (dummy here)
        """
        sdf_pred, log_var, latent_set = self.forward(seg_mask, coords)        
        # Compute target SDF from GT segmentation.
        target_sdf_full = compute_target_sdf(seg_mask)
        target_sdf_sampled = sample_target_sdf(target_sdf_full, coords)
        sdf_loss = sdf_reconstruction_loss(sdf_pred, target_sdf_sampled, log_var)
        kl_loss = 1e-3 * (latent_set**2).mean()
        return sdf_pred, sdf_loss, kl_loss

    def forward_return_logvar(self, seg_mask, coords):
        sdf_pred, log_var, latent_set = self.forward(seg_mask, coords)
        return sdf_pred, torch.tensor(0., device=seg_mask.device), torch.tensor(0., device=seg_mask.device), log_var

    def optimize_latents(self, seg_mask, coords, init_latent_set=None, num_iterations=100, lr=1e-2):
        """
        Inference phase: Given an imperfect segmentation, optimize the latent set (with network fixed)
        to best represent the segmentation.
        Returns optimized latent_set, sdf_pred, and log_var.
        """
        if init_latent_set is None:
            with torch.no_grad():
                init_latent_set = self.encoder(seg_mask)
        latent_set = init_latent_set.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latent_set], lr=lr)
        target_sdf = compute_target_sdf(seg_mask)
        target_sdf_sampled = torch.zeros((latent_set.shape[0], coords.shape[1]), device=seg_mask.device)
        for i in range(num_iterations):
            sdf_pred, log_var = self.decoder(latent_set, coords)
            loss = sdf_reconstruction_loss(sdf_pred, target_sdf_sampled, log_var) + 1e-3 * (latent_set**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f"Latent optimization iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")
        return latent_set, sdf_pred, log_var

#############################################
# Loss Functions
#############################################
def sdf_reconstruction_loss(sdf_pred, sdf_target, log_var):
    loss = 0.5 * torch.exp(-log_var) * (sdf_target - sdf_pred)**2 + 0.5 * log_var
    return loss.mean()

#############################################
# Visualization
#############################################
def visualize_uncertainty_3d(coords, log_var, sample_batch_idx=0, save_path=None):
    """
    Creates a 3D scatter plot of the uncertainty (log variance) values at the sampled coordinates
    for a given batch element.
    
    If save_path is provided, the figure is saved to that path and then closed.
    Otherwise, the figure is displayed interactively (useful with %matplotlib ipympl).
    
    Args:
      coords: torch.Tensor of shape [B, N, 3] with spatial coordinates.
      log_var: torch.Tensor of shape [B, N] with corresponding log variance values.
      sample_batch_idx: index of the batch element to visualize.
      save_path: If provided, the path to save the figure; if None, the plot is shown interactively.
    """
    xyz = coords[sample_batch_idx].detach().cpu().numpy()
    lv = log_var[sample_batch_idx].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=lv, cmap='jet', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='log variance')
    ax.set_title("Uncertainty (log_var) Scatter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def save_mesh_as_obj(verts, faces, filename):
    """
    Save the mesh (vertices and faces) to an OBJ file.
    """
    with open(filename, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ indices are 1-based.
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def save_reconstructed_mesh(geo_model, seg_mask, latent_set, output_resolution=(128,128,128), iso_level=0.0, save_path="reconstructed_mesh.obj"):
    """
    Generates a dense grid of coordinates over the volume defined by seg_mask,
    uses the decoder with the provided latent set to predict SDF values,
    applies marching cubes to extract the surface (at iso_level), and saves the
    resulting mesh as an OBJ file.
    
    Args:
      geo_model: GeometricLatentSetModel instance.
      seg_mask: torch.Tensor of shape [1, 1, H, W, D] (ground truth segmentation).
      latent_set: tensor of shape [1, M, latent_dim] produced by the encoder.
      output_resolution: tuple (H_r, W_r, D_r) for the reconstruction grid.
      iso_level: iso-surface value for marching cubes (commonly 0).
      save_path: file path to save the OBJ file.
    """
    B, C, H, W, D = seg_mask.shape
    H_r, W_r, D_r = output_resolution
    xs = torch.linspace(0, H-1, H_r)
    ys = torch.linspace(0, W-1, W_r)
    zs = torch.linspace(0, D-1, D_r)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)  # shape [H_r, W_r, D_r, 3]
    grid = grid.view(1, -1, 3).to(seg_mask.device)  # shape [1, num_points, 3]
    
    # Use the decoder with the provided latent set to predict SDF values on the dense grid.
    sdf_pred, _ = geo_model.decoder(latent_set, grid)  # [1, num_points]
    sdf_pred = sdf_pred.view(H_r, W_r, D_r).detach().cpu().numpy()
    
    # Check the range of the SDF volume.
    min_val, max_val = sdf_pred.min(), sdf_pred.max()
    print("Predicted SDF volume stats:", sdf_pred.min(), sdf_pred.max())
    if not (min_val <= iso_level <= max_val):
        iso_level = (min_val + max_val) / 2.0
        print(f"Warning: Desired iso_level not in range [{min_val:.4f}, {max_val:.4f}]. Using iso_level={iso_level:.4f}")

    # Apply marching cubes to extract the mesh.
    verts, faces, normals, values = measure.marching_cubes(sdf_pred, level=iso_level, spacing=(H / H_r, W / W_r, D / D_r))
    
    save_mesh_as_obj(verts, faces, save_path)
    print(f"Reconstructed mesh saved as {save_path}")
