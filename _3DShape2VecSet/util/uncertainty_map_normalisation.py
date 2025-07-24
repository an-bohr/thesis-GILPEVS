"""
Compute per-channel normalization stats (0.5/99.5 percentile, mean, std)
for every _XXXX.nii.gz in a training folder, then apply the same
clip→zero-mean→unit-std normalization to each channel in a target folder.
"""

import os
import re
import glob
import json
import argparse

import numpy as np
import nibabel as nib

CHANNEL_RE = re.compile(r'_(\d{4})\.nii\.gz$')

def gather_channel_files(folder, pattern='*_****.nii.gz'):
    """
    Return a dict mapping channel_id (e.g. '0001') -> list of file paths.
    """
    files = glob.glob(os.path.join(folder, pattern))
    channels = {}
    for fn in files:
        m = CHANNEL_RE.search(fn)
        if not m:
            continue
        ch = m.group(1)
        channels.setdefault(ch, []).append(fn)
    return channels

def compute_stats_per_channel(train_dir):
    """
    Scan train_dir, group files by channel suffix, compute stats for each.
    Returns dict: { channel: {mean,std, min, max}, ... }
    """
    channels = gather_channel_files(train_dir)
    if not channels:
        raise FileNotFoundError(f"No channelized NIfTIs found in {train_dir}")
    stats = {}
    for ch, paths in channels.items():
        # accumulate all voxels
        all_vals = []
        print(f"[INFO] computing stats for channel {ch} ({len(paths)} files)...")
        for fn in paths:
            data = nib.load(fn).get_fdata().ravel()
            all_vals.append(data)
        allv = np.concatenate(all_vals)
        stats[ch] = {
            'mean':  float(allv.mean()),
            'std':   float(allv.std()),
            'min':   float(allv.min()),
            'max':   float(allv.max())
        }
    return stats

def apply_normalization(train_dir, stats, pattern='*_****.nii.gz'):
    """
    For each NIfTI in train_dir matching pattern,
    parse its channel suffix, look up stats, and normalize in-place.
    """
    files = glob.glob(os.path.join(train_dir, '**', pattern), recursive=True)
    if not files:
        print(f"[WARNING] No files matching {pattern} in {train_dir}")
        return
    for fn in files:
        m = CHANNEL_RE.search(fn)
        if not m:
            print(f"[SKIP] filename does not match channel pattern: {fn}")
            continue
        ch = m.group(1)
        if ch not in stats:
            print(f"[SKIP] no stats for channel {ch}, file {fn}")
            continue

        s = stats[ch]
        img  = nib.load(fn)
        data = img.get_fdata().astype(np.float32)

        # zero-mean, unit-std
        data -= s['mean']
        data /= max(s['std'], 1e-8)

        nib.save(nib.Nifti1Image(data, img.affine, img.header), fn)
        # print(f"[NORM] channel {ch}: normalized {fn}")

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--train_dir', required=True, help='Dir with training volumes to compute stats and normalise in-place')
    p.add_argument('--stats_out', help='Path to write computed stats JSON')
    args = p.parse_args()

    stats = compute_stats_per_channel(args.train_dir)
    print("[INFO] computed per-channel stats:")
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        with open(args.stats_out, 'w') as f:
            json.dump(stats, f, indent=2)
        # print(f"[INFO] wrote stats to {args.stats_out}")

    apply_normalization(args.train_dir, stats)

if __name__ == '__main__':
    main()
