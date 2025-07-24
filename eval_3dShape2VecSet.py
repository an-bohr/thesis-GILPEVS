import os
import numpy as np
import nibabel as nib
import torch.backends.cudnn as cudnn
import torch
import argparse

import _3DShape2VecSet.util.misc as misc
from _3DShape2VecSet.util.dataset_topcow import TopCowDataset
import _3DShape2VecSet.models_ae as models_ae
from torch.serialization import add_safe_globals
from nibabel.processing import resample_from_to


def embed_and_crop_recon(mean_vol: np.ndarray,
                         sigma_vol: np.ndarray,
                         cta_img: nib.Nifti1Image
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    1) Map recon volumes into CTA space (re-affine + resample)
    2) Crop voxels outside the maximal GT-centroid radius
    Returns: (mean_cta, sigma_cta, affine, header)
    """
    orig_affine = cta_img.affine.copy()
    nx, ny, nz = cta_img.shape
    D = mean_vol.shape[0] - 1

    # compute scaling to align with CTA grid
    scale = [(nx - 1)/D, (ny - 1)/D, (nz - 1)/D]
    recon_aff = np.eye(4)
    recon_aff[:3,:3] = orig_affine[:3,:3] @ np.diag(scale)
    recon_aff[:3,3] = orig_affine[:3,3]

    # wrap volumes
    mean_img  = nib.Nifti1Image(mean_vol.astype(np.float32), recon_aff)
    sigma_img = nib.Nifti1Image(sigma_vol.astype(np.float32), recon_aff)

    # resample into CTA grid
    mean_cta = resample_from_to(mean_img,  cta_img, order=1,
                                mode='constant', cval=mean_vol.max()+1.0).get_fdata()
    sigma_cta = resample_from_to(sigma_img, cta_img, order=1,
                                 mode='constant', cval=0.0).get_fdata()

    return mean_cta, sigma_cta, orig_affine, cta_img.header.copy()

parser = argparse.ArgumentParser()
parser.add_argument('--model',       required=True, help='Model name')
parser.add_argument('--pth',         required=True, help='Checkpoint path')
parser.add_argument('--device',      default='cuda', help='Torch device')
parser.add_argument('--seed',        type=int, default=0, help='Random seed')
parser.add_argument('--data_path',   required=True, help='TopCoW data dir (all .nii.gz files)')
parser.add_argument('--cta_img_dir', required=True, help='Original CTA images dir (0000)')
parser.add_argument('--gt_mask_dir', required=True, help='GT mask images dir')
parser.add_argument('--output_dir',  default='predictions', help='Output NIfTI dir')
args = parser.parse_args()


def main():
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # load trained model
    model = models_ae.__dict__[args.model]()
    add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(args.pth, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(args.device).eval()

    # build SDF grid
    D = 128
    x = np.linspace(-1,1,D+1, dtype=np.float32)
    xv, yv, zv = np.meshgrid(x,x,x, indexing='ij')
    pts = np.stack([xv,yv,zv], axis=-1).reshape(-1,3)
    grid = torch.from_numpy(pts).unsqueeze(0).to(args.device)

    # data loader: no split, all NIfTI files
    ds = TopCowDataset(nifti_dir=args.data_path,
                       transform=None,
                       num_samples=2048)
    loader = torch.utils.data.DataLoader(ds,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8)

    os.makedirs(args.output_dir, exist_ok=True)
    metric_logger = misc.MetricLogger(delimiter='  ')

    for _, _, surface, _, fnames in metric_logger.log_every(loader, 10, 'Test:'):
        case_id = fnames[0][:-7]
        idxs = torch.randperm(surface.shape[1])[:2048]
        surf = surface[:, idxs, :].to(args.device)

        # forward pass
        out = model(surf, grid)
        mean_vol  = out['logits'][0].view(D+1,D+1,D+1).detach().cpu().numpy()
        logvar_vol= out['logvar'][0].view(D+1,D+1,D+1).detach().cpu().numpy()
        sigma_vol = np.exp(0.5 * logvar_vol)

        # load CTA and GT mask
        cta_img  = nib.load(os.path.join(args.cta_img_dir, f"{case_id}_0000.nii.gz"))
        mask_img = nib.load(os.path.join(args.gt_mask_dir, f"{case_id}.nii.gz"))

        # resample & crop
        mean_cta, sigma_cta, affine, header = embed_and_crop_recon(mean_vol, sigma_vol, cta_img)

        # threshold for binary
        bin_cta = (mean_cta <= 0.0).astype(np.uint8)

        # sample sparse SDF channel
        EPS, DELTA = 0.005, 0.10
        N_SURF, N_AROUND = 25000, 25000
        abs_sdf = np.abs(mean_cta)
        surf_idx = np.column_stack(np.nonzero(abs_sdf <= EPS))
        ext_idx  = np.column_stack(np.nonzero((mean_cta > EPS) & (abs_sdf <= DELTA)))
        int_idx  = np.column_stack(np.nonzero((mean_cta < -EPS) & (abs_sdf <= DELTA)))

        def sample(idxs, n):
            if idxs.shape[0] == 0:
                return np.zeros((0,3), int)
            replace = idxs.shape[0] < n
            choice = np.random.choice(idxs.shape[0], n, replace=replace)
            return idxs[choice]

        pts = np.vstack([sample(surf_idx, N_SURF),
                         sample(ext_idx, N_AROUND//2),
                         sample(int_idx, N_AROUND//2)])
        unique_pts = np.unique(pts, axis=0)

        # crop sigma outside bbox
        mask_data = mask_img.get_fdata() > 0
        coords = np.array(np.nonzero(mask_data))
        pad = 20
        i0,i1 = coords[0].min(), coords[0].max()
        j0,j1 = coords[1].min(), coords[1].max()
        k0,k1 = coords[2].min(), coords[2].max()
        i0p, i1p = max(0,i0-pad), min(mean_cta.shape[0]-1,i1+pad)
        j0p, j1p = max(0,j0-pad), min(mean_cta.shape[1]-1,j1+pad)
        k0p, k1p = max(0,k0-pad), min(mean_cta.shape[2]-1,k1+pad)
        bbox = np.zeros_like(bin_cta, bool)
        bbox[i0p:i1p+1, j0p:j1p+1, k0p:k1p+1] = True
        sigma_cta[~bbox] = 0

        # rasterize sparse channel
        pointcloud_vol = np.zeros_like(mean_cta, dtype=np.float32)
        for (i,j,k) in unique_pts:
            pointcloud_vol[i,j,k] = mean_cta[i,j,k]

        # save outputs
        nib.save(nib.Nifti1Image(sigma_cta, affine, header),
                 os.path.join(args.output_dir, f"{case_id}_0001.nii.gz"))
        nib.save(nib.Nifti1Image(pointcloud_vol, affine, header),
                 os.path.join(args.output_dir, f"{case_id}_0002.nii.gz"))

        print(f"Done case {case_id}")

    print("All done.")

if __name__ == '__main__':
    main()