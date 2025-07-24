# """
# refine_pipeline.py

# Looped refinement:
#   1) Geo-model inference (uncertainty + distance maps) via eval.py
#   2) Post-processing (clip→zero-mean→unit-std) via uncertainty_map_normalisation.py
#   3) Assemble 2- or 3-channel input (CTA + σ [ + distance ]) for nnU-Net
#   4) nnU-Net inference via srun nnUNetv2_predict
#   5) Feed refined masks back into step 1, for N iterations
# """

# import argparse, subprocess, os, shutil, sys

# def run_geo_inference(args, in_masks, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     # cmd = [
#     #     'srun', 'python', args.geo_script,
#     #     '--pth',          args.geo_ckpt,
#     #     '--model',        args.geo_model_name,
#     #     '--device',       args.geo_device,
#     #     '--data_path',    args.data_path,
#     #     '--cta_img_dir',  args.cta_dir,
#     #     '--gt_mask_dir',  in_masks,
#     #     '--output_dir',   out_dir
#     # ]
#     # Uncomment below and comment above if not running on Snellius:
#     cmd = [
#         'python', args.geo_script,
#         '--pth',          args.geo_ckpt,
#         '--model',        args.geo_model_name,
#         '--device',       args.geo_device,
#         '--data_path',    args.data_path,
#         '--cta_img_dir',  args.cta_dir,
#         '--gt_mask_dir',  in_masks,
#         '--output_dir',   out_dir
#     ]
#     print(">>> Running geo inference:", " ".join(cmd))
#     subprocess.run(cmd, check=True)

# def postprocess_geo_output(args, geo_out):
#     # cmd = [
#     #     'srun', 'python', args.postproc_script,
#     #     '--train_dir', geo_out
#     # ]
#     # Uncomment below and comment above if not running on Snellius:
#     cmd = [
#         'python', args.postproc_script,
#         '--train_dir', geo_out
#     ]
#     print(">>> Post-processing geo maps:", " ".join(cmd))
#     subprocess.run(cmd, check=True)

# def prepare_nnunet_input(cta_dir, geo_out, nn_input, use_distance):
#     if os.path.exists(nn_input):
#         shutil.rmtree(nn_input)
#     os.makedirs(nn_input, exist_ok=True)
#     for fn in os.listdir(cta_dir):
#         if not fn.endswith('_0000.nii.gz'): continue
#         case = fn[:-12]
#         # link CTA
#         os.symlink(
#             os.path.join(cta_dir, fn),
#             os.path.join(nn_input, f"{case}_0000.nii.gz")
#         )
#         # link uncertainty (_0001)
#         unc = os.path.join(geo_out, f"{case}_0001.nii.gz")
#         os.symlink(unc, os.path.join(nn_input, f"{case}_0001.nii.gz"))
#         # optionally link distance (_0002)
#         if use_distance:
#             dist = os.path.join(geo_out, f"{case}_0002.nii.gz")
#             os.symlink(dist, os.path.join(nn_input, f"{case}_0002.nii.gz"))

# def run_nnunet(args, nn_input, nn_output):
#     os.makedirs(nn_output, exist_ok=True)
#     # cmd = [
#     #     'srun', 'nnUNetv2_predict',
#     #     '-i',  nn_input,
#     #     '-o',  nn_output,
#     #     '-d',  args.nnunet_task,
#     #     '-c',  args.nnunet_config,
#     #     '-chk',args.nnunet_checkpoint
#     # ]
#     # Uncomment below and comment above if not running on Snellius:
#     cmd = [
#         'nnUNetv2_predict',
#         '-i',  nn_input,
#         '-o',  nn_output,
#         '-d',  args.nnunet_task,
#         '-c',  args.nnunet_config,
#         '-chk',args.nnunet_checkpoint
#     ]
#     print(">>> Running nnU-Net:", " ".join(cmd))
#     subprocess.run(cmd, check=True)

# def main():
#     p = argparse.ArgumentParser(__doc__)
#     # common inputs
#     p.add_argument('--cta_dir',       required=True,
#                    help='Folder of your CTA *_0000.nii.gz files')
#     p.add_argument('--initial_masks', required=True,
#                    help='Seed masks (imperfect nnU-Net outputs)')
#     p.add_argument('--data_path',     required=True,
#                    help='TopCoW data dir (for eval.py)')
#     # geo‐model
#     p.add_argument('--geo_script',        default='eval.py',
#                    help='Path to your eval.py')
#     p.add_argument('--geo_ckpt',          required=True,
#                    help='geo-model checkpoint .pth')
#     p.add_argument('--geo_model_name',    default='ae_d64_256_depth6_outdim2',
#                    help='(optional) model name arg to eval.py')
#     p.add_argument('--geo_device',        default='cuda',
#                    help='torch device for eval.py')
#     # post-processing
#     p.add_argument('--postproc_script', default='uncertainty_map_normalisation.py',
#                    help='Path to your post-processing script')
#     # nnU-Net
#     p.add_argument('--nnunet_task',      required=True,
#                    help='nnUNet task ID (e.g. "016")')
#     p.add_argument('--nnunet_config',    default='3d_fullres',
#                    help='nnUNet config: "2d" OR "3d_fullres"')
#     p.add_argument('--nnunet_checkpoint',required=True,
#                    help='checkpoint_best.pth OR checkpoint_final.pth for nnU-Net')
#     p.add_argument('--use_distance',     action='store_true',
#                    help='Include distance (_0002) as 3rd channel')
#     # loop & dirs
#     p.add_argument('--iterations', type=int, default=1,
#                    help='How many refinement loops')
#     p.add_argument('--work_dir',    default='work',
#                    help='Base output dir for all iterations')
#     args = p.parse_args()

#     current_masks = args.initial_masks

#     for i in range(1, args.iterations + 1):
#         print(f"\n=== ITERATION {i}/{args.iterations} ===")
#         it_dir    = os.path.join(args.work_dir, f"iter{i}")
#         geo_out   = os.path.join(it_dir, "geo_out")
#         nn_input  = os.path.join(it_dir, "nnUNet_input")
#         nn_output = os.path.join(it_dir, "nnUNet_output")

#         # 1) geo inference → geo_out/
#         run_geo_inference(args, current_masks, geo_out)

#         # 2) post-process both _0000 & _0001 in-place
#         postprocess_geo_output(args, geo_out)

#         # 3) assemble CTA + σ [+ distance] → nnUNet_input/
#         prepare_nnunet_input(args.cta_dir, geo_out, nn_input, args.use_distance)

#         # 4) nnU-Net prediction → nnUNet_output/
#         run_nnunet(args, nn_input, nn_output)

#         # 5) feed refined masks into next iteration
#         current_masks = nn_output

#     print("\n🎉 Refinement done. Final masks in:", current_masks)

# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""
refine_pipeline.py

Looped refinement (sequential on a single allocated node):
  1) Geo-model inference via eval_3dShape2VecSet.py
  2) Post-processing via uncertainty_map_normalisation.py
  3) Assemble 2- or 3-channel input (CTA + uncertainty [ + distance ]) for nnU-Net
  4) nn-U-Net inference via nnUNetv2_predict
  5) Loop for N iterations, feeding back refined masks

Invoke with a single srun allocation:
  srun python refine_pipeline.py [args]
"""

import argparse
import subprocess
import os
import shutil
import sys


def run_geo_inference(args, in_masks, out_dir):
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # prepare a CTA-only folder containing only *_0000.nii.gz
    base_dir = os.path.dirname(out_dir)
    cta_only_dir = os.path.join(base_dir, "cta_only")
    if os.path.exists(cta_only_dir):
        shutil.rmtree(cta_only_dir)
    os.makedirs(cta_only_dir, exist_ok=True)
    for fname in os.listdir(args.cta_dir):
        if fname.endswith('_0000.nii.gz'):
            src = os.path.join(args.cta_dir, fname)
            dst = os.path.join(cta_only_dir, fname)
            os.symlink(src, dst)

    # run geo inference using CTA-only dir
    cmd = [
        sys.executable, args.geo_script,
        '--pth',         args.geo_ckpt,
        '--model',       args.geo_model_name,
        '--device',      args.geo_device,
        '--data_path',   args.data_path,
        '--cta_img_dir', cta_only_dir,
        '--gt_mask_dir', in_masks,
        '--output_dir',  out_dir
    ]
    print(">>> Running geo inference:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def postprocess_geo_output(args, geo_out):
    cmd = [
        sys.executable, args.postproc_script,
        '--train_dir', geo_out
    ]
    print(">>> Post-processing geo maps:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepare_nnunet_input(cta_dir, geo_out, nn_input, use_distance):
    if os.path.exists(nn_input):
        shutil.rmtree(nn_input)
    os.makedirs(nn_input, exist_ok=True)
    for fn in os.listdir(cta_dir):
        if not fn.endswith('_0000.nii.gz'):
            continue
        case = fn[:-12]
        # link CTA
        os.symlink(
            os.path.join(cta_dir, fn),
            os.path.join(nn_input, f"{case}_0000.nii.gz")
        )
        # link uncertainty (_0001)
        unc = os.path.join(geo_out, f"{case}_0001.nii.gz")
        os.symlink(unc, os.path.join(nn_input, f"{case}_0001.nii.gz"))
        # optionally link distance (_0002)
        if use_distance:
            dist = os.path.join(geo_out, f"{case}_0002.nii.gz")
            os.symlink(dist, os.path.join(nn_input, f"{case}_0002.nii.gz"))


def run_nnunet(args, nn_input, nn_output):
    os.makedirs(nn_output, exist_ok=True)
    cmd = [
        'nnUNetv2_predict',
        '-i', nn_input,
        '-o', nn_output,
        '-d', args.nnunet_task,
        '-c', args.nnunet_config,
        '-chk', args.nnunet_checkpoint,
        '--disable_progress_bar'
    ]
    print(">>> Running nnU-Net:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    # CTA + masks + data
    p.add_argument('--cta_dir',       required=True, help='Folder of CTA *_0000.nii.gz files')
    p.add_argument('--initial_masks', required=True, help='Seed masks (imperfect nnU-Net outputs)')
    p.add_argument('--data_path',     required=True, help='TopCoW data dir')
    # Geo-model settings
    p.add_argument('--geo_script',     default='eval_3dShape2VecSet.py', help='Path to geo inference script')
    p.add_argument('--geo_ckpt',       required=True, help='Geo-model checkpoint .pth')
    p.add_argument('--geo_model_name', default='ae_d64_256_depth6_outdim2', help='Model name')
    p.add_argument('--geo_device',     default='cuda', help='Device for geo inference')
    # Post-processing
    p.add_argument('--postproc_script', default='uncertainty_map_normalisation.py', help='Path to post-processing script')
    # nnU-Net settings
    p.add_argument('--nnunet_task',      required=True, help='nnUNet task ID (e.g. 016)')
    p.add_argument('--nnunet_config',    default='3d_fullres', help='nnUNet config')
    p.add_argument('--nnunet_checkpoint',required=True, help='nnUNet checkpoint file')
    p.add_argument('--use_distance',     action='store_true', help='Include distance channel (3rd)')
    # Loop & directories
    p.add_argument('--iterations', type=int, default=1, help='Number of refinement loops')
    p.add_argument('--work_dir',    default='work', help='Base dir for iteration outputs')
    args = p.parse_args()

    current_masks = args.initial_masks

    for i in range(1, args.iterations + 1):
        print(f"\n=== ITERATION {i}/{args.iterations} ===")
        it_dir    = os.path.join(args.work_dir, f"iter{i}")
        geo_out   = os.path.join(it_dir, "geo_out")
        nn_input  = os.path.join(it_dir, "nnUNet_input")
        nn_output = os.path.join(it_dir, "nnUNet_output")

        # 1) Geo-model inference
        run_geo_inference(args, current_masks, geo_out)

        # 2) In-place post-processing
        postprocess_geo_output(args, geo_out)

        # 3) Assemble nnU-Net input
        prepare_nnunet_input(args.cta_dir, geo_out, nn_input, args.use_distance)

        # 4) nnU-Net inference
        run_nnunet(args, nn_input, nn_output)

        # 5) Next iteration uses refined masks
        current_masks = nn_output

    print("\n🎉 Refinement done. Final masks in:", current_masks)

if __name__ == '__main__':
    main()
