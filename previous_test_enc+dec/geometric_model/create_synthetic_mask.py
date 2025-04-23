import numpy as np
import nibabel as nib

#############################################
# Synthetic Mask Generation
#############################################
def create_synthetic_mask(shape=(64,64,64), sphere_center=None, sphere_radius=20):
    """
    Create a synthetic binary segmentation mask containing a sphere.
    
    Args:
      shape: tuple (H, W, D) of the volume.
      sphere_center: tuple (x, y, z) center of sphere; defaults to center of volume.
      sphere_radius: radius of the sphere.
    
    Returns:
      A numpy array of shape [H, W, D] with 1 inside the sphere and 0 outside.
    """
    if sphere_center is None:
        sphere_center = (shape[0] / 2, shape[1] / 2, shape[2] / 2)
    X, Y, Z = np.indices(shape)
    dist = np.sqrt((X - sphere_center[0])**2 +
                   (Y - sphere_center[1])**2 +
                   (Z - sphere_center[2])**2)
    mask = (dist < sphere_radius).astype(np.uint8)
    return mask

def save_synthetic_mask_nifti(mask, save_path="synthetic_mask.nii.gz"):
    """
    Saves a synthetic segmentation mask as a NIfTI file.
    """
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(mask, affine)
    nib.save(nii_img, save_path)
    print(f"Synthetic mask saved as {save_path}")