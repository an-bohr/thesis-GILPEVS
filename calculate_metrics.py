import os
import json
import argparse

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion
from skimage.morphology import skeletonize

def skeletonize_3d(arr):
    """
    Get 3D skeleton by applying 2D skeletonize on each axial slice.
    Input: binary numpy array
    Output: boolean numpy array
    """
    arr = arr.astype(bool)
    skel = np.zeros_like(arr, dtype=bool)
    # assume first axis is slice
    for i in range(arr.shape[0]):
        skel[i] = skeletonize(arr[i])
    return skel

def dice_coeff(bin_pred, bin_gt):
    """Dice = 2|A∩B|/(|A|+|B|)"""
    pred_f = bin_pred.astype(bool)
    gt_f   = bin_gt.astype(bool)
    inter = np.logical_and(pred_f, gt_f).sum()
    denom = pred_f.sum() + gt_f.sum()
    return 2.0 * inter / (denom + 1e-8)

def class_dices(lbl_pred, lbl_gt):
    """Per-class Dice for all nonzero labels, plus their mean."""
    labels = set(np.unique(lbl_gt)) | set(np.unique(lbl_pred))
    labels.discard(0)
    dices = {}
    for c in sorted(labels):
        dices[int(c)] = dice_coeff(lbl_pred==c, lbl_gt==c)
    mean_d = float(np.mean(list(dices.values()))) if dices else 0.0
    return dices, mean_d

def cldice(bin_pred, bin_gt):
    """
    clDice = 2·Tprec·Tsens/(Tprec+Tsens)
      where Tprec = |S(G)∩P|/|S(G)|, Tsens = |S(P)∩G|/|S(P)|.
    """
    G = bin_gt.astype(bool)
    P = bin_pred.astype(bool)
    Sg = skeletonize_3d(G)
    Sp = skeletonize_3d(P)
    tprec = np.logical_and(Sg, P).sum() / (Sg.sum() + 1e-8)
    tsens = np.logical_and(Sp, G).sum() / (Sp.sum() + 1e-8)
    return 2 * tprec * tsens / (tprec + tsens + 1e-8)

def hd95(bin_pred, bin_gt, spacing=(1.0,1.0,1.0)):
    """
    95th-percentile symmetric surface distance.
    """
    P = bin_pred.astype(bool)
    G = bin_gt.astype(bool)
    # get boundaries
    struct = np.ones((3,3,3), dtype=bool)
    BdP = np.logical_xor(P, binary_erosion(P, struct))
    BdG = np.logical_xor(G, binary_erosion(G, struct))
    # distance maps to nearest foreground of the other
    dtP = distance_transform_edt(~P, sampling=spacing)
    dtG = distance_transform_edt(~G, sampling=spacing)
    d1 = dtP[BdG]  # GT‐boundary → pred
    d2 = dtG[BdP]  # Pred‐boundary → gt
    all_d = np.concatenate([d1, d2])
    if all_d.size == 0:
        return 0.0
    return float(np.percentile(all_d, 95))

def vol_sim(bin_pred, bin_gt):
    """
    Volumetric similarity: 1 - |V_P - V_G|/(V_P + V_G).
    """
    vp = bin_pred.sum()
    vg = bin_gt.sum()
    if vp + vg == 0:
        return 1.0
    return 1.0 - abs(vp - vg) / float(vp + vg)

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.int32)
    spacing = img.header.get_zooms()[:3]
    return data, spacing

def main(pred_dir, gt_dir, out_json):
    results = {}
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith('.nii.gz'):
            continue
        case_id = fname.replace('.nii.gz','')
        ppath = os.path.join(pred_dir, fname)
        gpath = os.path.join(gt_dir,   fname)
        if not os.path.exists(gpath):
            print(f"[WARN] no GT for {fname}, skipping")
            continue

        pred, _ = load_nifti(ppath)
        gt, spacing = load_nifti(gpath)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {fname}: {pred.shape} vs {gt.shape}")

        bin_pred = pred > 0
        bin_gt = gt > 0

        # compute metrics
        b_dice = dice_coeff(bin_pred, bin_gt)
        cls_dict, mc  = class_dices(pred, gt)
        cl_dice = cldice(bin_pred, bin_gt)
        h95_val = hd95(bin_pred, bin_gt, spacing)
        vsim = vol_sim(bin_pred, bin_gt)

        results[case_id] = {
            "binary_dice": float(b_dice),
            "class_dice": {str(k): float(v) for k,v in cls_dict.items()},
            "mean_class_dice": float(mc),
            "cldice": float(cl_dice),
            "hausdorff95": float(h95_val),
            "volumetric_similarity": float(vsim)
        }

    # compute averages of each scalar metric
    if results:
        n = len(results)
        avg = {
            "binary_dice": sum(r["binary_dice"] for r in results.values()) / n,
            "mean_class_dice": sum(r["mean_class_dice"] for r in results.values()) / n,
            "cldice": sum(r["cldice"] for r in results.values()) / n,
            "hausdorff95": sum(r["hausdorff95"] for r in results.values()) / n,
            "volumetric_similarity":sum(r["volumetric_similarity"] for r in results.values()) / n,
        }
    else:
        avg = {}

    # build output with average at the top
    out = {"average_metrics": avg}
    out.update(results)

    # ensure parent directory for output exists
    out_dir = os.path.dirname(out_json)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ensure the JSON file itself exists
    from pathlib import Path
    out_path = Path(out_json)
    out_path.touch(exist_ok=True)

    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics summary (with averages) to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute segmentation metrics")
    parser.add_argument("--pred_dir", required=True, help="Directory of predicted NIfTI masks")
    parser.add_argument("--gt_dir", required=True, help="Directory of ground-truth NIfTI masks")
    parser.add_argument("--output", required=True, help="Path for JSON summary")
    args = parser.parse_args()
    main(args.pred_dir, args.gt_dir, args.output)