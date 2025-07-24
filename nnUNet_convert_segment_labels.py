import os
import numpy as np
import nibabel as nib
from glob import glob

# Change this to the directory containing your .nii.gz segmentation files
input_dir = "/nnUNet/nnunetv2/nnUNet_raw/Dataset011_TopCoW/labelsTr"
output_dir = "/nnUNet/nnunetv2/nnUNet_raw/Dataset011_TopCoW/labelsTr"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each NIfTI file in the input directory
for file_path in glob(os.path.join(input_dir, "*.nii.gz")):
    print(f"Processing: {file_path}")
    
    # Load the NIfTI image
    nifti_img = nib.load(file_path)
    segmentation_data = nifti_img.get_fdata().astype(np.uint16)  # Convert to integer format
    
    # Modify labels: Change 15 → 13
    segmentation_data[segmentation_data == 15] = 13

    # Create new NIfTI image with the modified segmentation
    new_nifti_img = nib.Nifti1Image(segmentation_data, affine=nifti_img.affine, header=nifti_img.header)
    
    # Save the modified file to the output directory
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    nib.save(new_nifti_img, output_path)
    
    print(f"Saved modified file: {output_path}")

print("Label conversion completed!")
