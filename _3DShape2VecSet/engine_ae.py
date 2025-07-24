import math
from typing import Iterable
import os
import matplotlib.pyplot as plt

import torch

import _3DShape2VecSet.util.misc as misc
import _3DShape2VecSet.util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    log_writer=None,
                    args=None):
    """
    Single training epoch with:
      - L1 reconstruction loss
      - Eikonal loss (|∇f| = 1 everywhere)
      - Surface-normal consistency loss (∇f at surface aligns with true normal)
      - Optional KL term if model returns 'kl'

    Args:
      model       : the auto-encoder SDF model
      criterion   : torch.nn.L1Loss(reduction='mean') or similar
      data_loader : yields (points, labels, surface_pts, surface_normals, _)
      optimizer   : optimizer
      device      : torch.device
      epoch       : current epoch index
      loss_scaler : for mixed-precision (can be a no-op if unused)
      max_norm    : gradient clipping norm
      log_writer  : TensorBoard writer or None
      args        : must contain nll_weight, accum_iter, etc.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('recon', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('nll', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # optional KL weight
    kl_weight = getattr(args, 'kl_weight', 1e-3)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, labels, surface_pts, surface_normals, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # adjust LR per iteration (if using a custom scheduler)
        if data_iter_step % accum_iter == 0:
            cur_iter = data_iter_step / len(data_loader) + epoch
            lr_sched.adjust_learning_rate(optimizer, cur_iter, args)

        # Move data to GPU (or CPU) and prepare for gradient
        points = points.to(device, non_blocking=True)               # shape: (B, 2N, 3)
        points.requires_grad_(True)
        labels = labels.to(device, non_blocking=True)               # shape: (B, 2N)
        surface_pts = surface_pts.to(device, non_blocking=True)     # shape: (B, N, 3)
        surface_pts.requires_grad_(True)
        surface_normals = surface_normals.to(device, non_blocking=True)  # shape: (B, N, 3)

        # —————————————————————————————————————————————————————————————
        # Forward‐pass: model may return a dict or a raw tensor
        outputs = model(surface_pts, points)

        # Extract a possible KL‐term
        if isinstance(outputs, dict):
            loss_kl = outputs.get('kl', None)

            # Try to find the SDF‐tensor under known keys
            if 'logits' in outputs:
                sdf_full = outputs['logits']
            elif 'sdf' in outputs:
                sdf_full = outputs['sdf']
            else:
                # fallback: pick the first Tensor in the dict
                for v in outputs.values():
                    if isinstance(v, torch.Tensor):
                        sdf_full = v
                        break
                else:
                    raise RuntimeError("Model returned a dict with no tensor-valued entries")
        else:
            loss_kl = None
            sdf_full = outputs
        # —————————————————————————————————————————————————————————————
        # Reconstruction loss on all query points + Gaussian NLL:
        pred_mean = outputs['logits']
        pred_logvar = outputs['logvar']
        nll = 0.5 * ((pred_mean - labels)**2 / torch.exp(pred_logvar) + pred_logvar + math.log(2 * math.pi))
        nll_loss = nll.mean()
        nll_weight = args.nll_weight
        # Make sure we can compute gradients wrt points for normal terms
        points.requires_grad_(True)
        recon_loss = criterion(sdf_full, labels)

        # —————————————————————————————————————————————————————————————
        # Total loss = recon + nll_weight*nll_loss + optional KL
        loss = recon_loss + nll_weight * nll_loss
        if loss_kl is not None:
            loss += kl_weight * loss_kl

        loss_value = loss.item()

        # —————————————————————————————————————————————————————————————
        # Backward & step (with gradient accumulation if enabled)
        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # —————————————————————————————————————————————————————————————
        metric_logger.update(recon=recon_loss.item())
        metric_logger.update(total=loss_value)
        metric_logger.update(nll=nll_loss.item())
        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        # Track current LR
        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        # Write to TensorBoard if requested
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/total', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss/recon', recon_loss.item(), epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch):
    criterion = torch.nn.L1Loss(reduction="mean")

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for i, (points, labels, surface_pts, _, _) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface_pts = surface_pts.to(device, non_blocking=True)

        # ─── per-batch POINT-CLOUD SDF DEBUG ─────────────────────────────
        if args.debug and i == 0 and epoch == 0:
            # grab numpy arrays
            pts_np = points[0].cpu().numpy()
            surf_np = surface_pts[0].cpu().numpy()
            sdf_np = labels[0].cpu().numpy()

            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')

            # query points colored by SDF
            sc = ax.scatter(
                pts_np[:,0], pts_np[:,1], pts_np[:,2],
                c=sdf_np, cmap='coolwarm', s=2, alpha=0.6, label='query pts'
            )
            # true surface in black
            ax.scatter(
                surf_np[:,0], surf_np[:,1], surf_np[:,2],
                c='k', s=4, label='surface pts'
            )
            ax.set_title(f"Debug SDF @ epoch {epoch}")
            ax.legend()
            fig.colorbar(sc, label='SDF value')

            dbg_fn = os.path.join(args.output_dir, f"debug_pts_epoch_{epoch}.png")
            fig.savefig(dbg_fn, dpi=150)
            plt.close(fig)
            print(f"[debug pts] saved to {dbg_fn}")

        # compute output
        with torch.amp.autocast(device_type='cuda', enabled=False):

            outputs = model(surface_pts, points)
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']

            loss = criterion(outputs, labels)

        threshold = 0

        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()

        metric_logger.update(L1=loss.item(), n=points.shape[0])

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.L1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}