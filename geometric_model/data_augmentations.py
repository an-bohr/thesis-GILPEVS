"""
CPU-parallel offline data augmentation:
 - Mirror each .nii/.nii.gz along a specified axis
 - Rotate by ±angle° around each of X, Y, Z

This uses multiprocessing.Pool to run augmentations in parallel.
"""
import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate as ndi_rotate
from multiprocessing import Pool, cpu_count

# which voxel‐array axes correspond to the physical rotation planes
PLANE_FOR_AXIS = {
    0: (1, 2),  # rotate about X ↔ in Y–Z plane
    1: (0, 2),  # rotate about Y ↔ in X–Z plane
    2: (0, 1),  # rotate about Z ↔ in X–Y plane
}
AXIS_LETTER = {0: "X", 1: "Y", 2: "Z"}

def augment_file(args):
    """
    Load one file, do mirror + rotations, save out.
    """
    in_path, out_dir, mirror_axis, angle = args
    img = nib.load(in_path)
    seg = img.get_fdata().astype(np.int16)
    base, ext = os.path.splitext(os.path.basename(in_path))
    if ext == ".gz" and base.endswith(".nii"):
        base = base[:-4]
        ext = ".nii.gz"

    # 1) Mirror
    seg_mir = np.flip(seg, axis=mirror_axis).astype(np.int16)
    out_mir = os.path.join(out_dir, f"{base}_mirrored{ext}")
    nib.save(nib.Nifti1Image(seg_mir, img.affine, img.header), out_mir)

    # 2) Rotations ±angle about X/Y/Z
    for ax in (0, 1, 2):
        plane = PLANE_FOR_AXIS[ax]
        letter = AXIS_LETTER[ax]
        for sign in (-1, 1):
            θ = sign * angle
            seg_rot = ndi_rotate(
                seg,
                θ,
                axes=plane,
                reshape=False,
                order=0,          # nearest‐neighbor to keep integer labels
                mode="constant",
                cval=0
            ).astype(np.int16)
            out_rot = os.path.join(out_dir, f"{base}_rot{letter}{θ:+g}{ext}")
            nib.save(nib.Nifti1Image(seg_rot, img.affine, img.header), out_rot)

    return in_path  # for simple logging

def main():
    p = argparse.ArgumentParser(
        description="CPU-parallel mirror + rotate NIfTI segmentations")
    p.add_argument("-i", "--input_dir",  required=True,
                   help="Folder with original .nii/.nii.gz masks")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Where to save augmented masks")
    p.add_argument("--mirror_axis", type=int, default=0,
                   help="Array axis to mirror on (e.g. 0 = left↔right)")
    p.add_argument("--angle", type=float, default=10.0,
                   help="Max absolute rotation (in degrees) for each axis")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = [f for f in os.listdir(args.input_dir)
             if f.lower().endswith((".nii", ".nii.gz"))]
    if not files:
        print("No NIfTI files found in", args.input_dir)
        return

    # build argument list
    tasks = [
        (os.path.join(args.input_dir, f),
         args.output_dir,
         args.mirror_axis,
         args.angle)
        for f in files
    ]

    n_procs = min(cpu_count(), len(tasks))
    print(f"Found {len(tasks)} files. Spawning {n_procs} parallel workers.")
    with Pool(n_procs) as pool:
        for in_path in pool.imap_unordered(augment_file, tasks):
            print("Augmented:", os.path.basename(in_path))

    print("All done! Augmented data in", args.output_dir)

if __name__ == "__main__":
    main()
