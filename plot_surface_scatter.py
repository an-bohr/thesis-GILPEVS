import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_npz_samples(npz_file):
    data = np.load(npz_file)
    # "pos" should be [N, 3] (the 3D coordinates)
    # "neg" should be [N] (the corresponding SDF values)
    return data['pos'], data['neg']

def plot_full_sdf_scatter(coords, sdf_values, save_path='sdf_scatter_full.png'):
    """
    Scatter plot of all sample points color-coded by their SDF value.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # You might want to adjust marker size (s) or alpha
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=sdf_values,
                    cmap='coolwarm', marker='o', s=5, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='SDF value')
    ax.set_title("3D Scatter Plot of SDF Samples")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved full SDF scatter plot to {save_path}")

def plot_surface_scatter(coords, sdf_values, threshold=2.0, save_path='sdf_scatter_surface.png'):
    """
    Scatter plot showing only points in a narrow band around zero (the shape boundary).
    """
    # Select only those points with |SDF| smaller than the threshold:
    mask = np.abs(sdf_values) < threshold
    surface_coords = coords[mask]
    surface_sdf = sdf_values[mask]
    
    if surface_coords.size == 0:
        print("No points found near the surface with the given threshold.")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(surface_coords[:, 0], surface_coords[:, 1], surface_coords[:, 2],
                    c=surface_sdf, cmap='coolwarm', marker='o', s=10)
    plt.colorbar(sc, ax=ax, label='SDF value (near zero)')
    ax.set_title(f"3D Scatter Plot of Surface Samples (|SDF| < {threshold})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved surface scatter plot to {save_path}")

if __name__ == "__main__":
    npz_path = "/home/lucasp/thesis/npz_output/topcow_ct_001.npz"
    coords, sdf_values = load_npz_samples(npz_path)

    # Create a full scatter plot color-coded by SDF value.
    plot_full_sdf_scatter(coords, sdf_values, save_path='sdf_scatter_full.png')

    # Create a scatter plot for points near the boundary.
    plot_surface_scatter(coords, sdf_values, threshold=2.0, save_path='sdf_scatter_surface.png')
