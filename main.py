import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time

from geometric_model.deep_sdf_decoder import Decoder
from geometric_model.utils import load_nifti_as_tensor, visualize_sdf_slice, plot_loss_curve, test_latent_sensitivity, plot_latent_distribution

# Custom dataset: one shape per npz file.
class CowSDFDataset(Dataset):
    def __init__(self, npz_folder, mapping_json, debug=False):
        self.npz_folder = npz_folder
        with open(mapping_json, "r") as f:
            self.mapping = json.load(f)
        self.shape_names = sorted(self.mapping.keys())
        self.debug = debug

    def __len__(self):
        return len(self.shape_names)

    def __getitem__(self, idx):
        shape_name = self.shape_names[idx]
        npz_path = os.path.join(self.npz_folder, shape_name + ".npz")
        data = np.load(npz_path)
        # Convert to tensors
        coords = torch.from_numpy(data["pos"]).float()  # [N, 3]
        sdf_vals = torch.from_numpy(data["neg"]).float()  # [N]
        if self.debug:
            print(f"Shape '{shape_name}': SDF min = {sdf_vals.min().item():.4f}, max = {sdf_vals.max().item():.4f}")
        shape_idx = self.mapping[shape_name]
        return shape_idx, coords, sdf_vals

def main():
    parser = argparse.ArgumentParser(description="Train or visualize DeepSDF auto-decoder for vessel segmentation.")
    parser.add_argument("--npz_folder", type=str, default="/home/lucasp/thesis/npz_output",
                        help="Folder containing npz SDF sample files and mapping.json.")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--latent_size", type=int, default=256,
                        help="Dimension of the latent codes.")
    parser.add_argument("--hidden_dims", type=str, default="512,512,512,512,512,512,512,512",
                        help="Comma-separated list of hidden dimensions for the decoder.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training.")
    parser.add_argument("--latent_lr_factor", type=float, default=0.1,
                        help="Multiplier for latent codes learning rate relative to the decoder.")
    parser.add_argument("--batch_split", type=int, default=1,
                        help="Number of sub-batches to split each batch into (for gradient accumulation).")
    parser.add_argument("--clamp_dist", type=float, default=1.0,
                        help="Clamping distance for SDF values.")
    parser.add_argument("--do_code_regularization", action="store_true",
                        help="If set, add latent code regularization to the loss.")
    parser.add_argument("--code_reg_lambda", type=float, default=1e-4,
                        help="Weight of latent code regularization term.")
    parser.add_argument("--visualize_only", action="store_true",
                        help="If set, only run visualization using the trained model; do not train.")
    parser.add_argument("--segmentation_file", type=str, default=None,
                        help="Path to a ground truth segmentation NIfTI file for visualization.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, print additional debug information.")
    parser.add_argument("--checkpoint", "-c", type=str, default="auto_decoder.pth",
                        help="Path to trained decoder checkpoint for visualization")

    args = parser.parse_args()
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    mapping_json = os.path.join(args.npz_folder, "mapping.json")

    # Visualization-only mode
    if args.visualize_only:
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        # Instantiate decoder exactly as during training.
        decoder = Decoder(
            latent_size=args.latent_size,
            dims=hidden_dims,
            dropout=[i for i in range(len(hidden_dims))],
            dropout_prob=0.2,
            norm_layers=[i for i in range(len(hidden_dims))],
            latent_in=[4],
            weight_norm=True,
            xyz_in_all=False,
            use_tanh=False,
        ).cuda()
        decoder.load_state_dict(checkpoint["decoder_state"])
        decoder.eval()
        n_shapes = len(checkpoint["latents_state"]["weight"])
        lat_vecs = torch.nn.Embedding(n_shapes, args.latent_size).cuda()
        lat_vecs.load_state_dict(checkpoint["latents_state"])
        print("Loaded trained model for visualization.")

        if args.segmentation_file:
            seg_mask = load_nifti_as_tensor(args.segmentation_file)
            latent_code = lat_vecs(torch.tensor([0]).cuda())
            visualize_sdf_slice(decoder, latent_code, seg_mask, slice_idx=None,
                                 device="cuda", save_path="sdf_slice.png")
            print("Visualization saved as sdf_slice.png")
        else:
            print("No segmentation file provided for visualization.")
        return

    # ---------------------- Training Mode ----------------------
    dataset = CowSDFDataset(args.npz_folder, mapping_json, debug=args.debug)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    n_shapes = len(dataset)
    print(f"Found {n_shapes} shapes in dataset.")

    decoder = Decoder(
        latent_size=args.latent_size,
        dims=hidden_dims,
        dropout=[i for i in range(len(hidden_dims))],
        dropout_prob=0.2,
        norm_layers=[i for i in range(len(hidden_dims))],
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
    ).cuda()

    # Create latent embedding with improved initialization.
    lat_vecs = nn.Embedding(n_shapes, args.latent_size)
    init_std = 1.0 / (args.latent_size ** 0.5)
    torch.nn.init.normal_(lat_vecs.weight, 0.0, init_std)
    lat_vecs = lat_vecs.cuda()

    # Set separate learning rates: lower for latent codes.
    decoder_lr = args.learning_rate
    latent_lr = args.learning_rate * args.latent_lr_factor
    optimizer = optim.Adam([
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': lat_vecs.parameters(), 'lr': latent_lr}
    ])
    loss_fn = nn.L1Loss(reduction="sum")

    enforce_minmax = True
    minT = -args.clamp_dist
    maxT = args.clamp_dist
    batch_split = args.batch_split

    iteration_loss_log = []  # every shape/batch
    epoch_loss_log = []      # average per epoch
    best_loss = float('inf') # for saving best model

    # (Optional) Debug: Check overall SDF range for each shape before training.
    if args.debug:
        print("DEBUG: Checking SDF ranges for all shapes in the dataset:")
        for i in range(len(dataset)):
            shape_idx, coords, sdf_vals = dataset[i]
            print(f"Shape {dataset.shape_names[i]}: SDF min = {sdf_vals.min().item():.4f}, max = {sdf_vals.max().item():.4f}")

    # Added to test speedups with amp 
    # NOTE: doesn't seem to be that good for reconstruction even tho it does reduce training time by around 1.5
    # scaler = torch.amp.GradScaler()
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        decoder.train()
        epoch_loss_sum = 0.0
        epoch_num_batches = 0
        epoch_loss_values = []

        for shape_idx, coords, sdf_vals in dataloader:
            sdf_data = torch.cat([coords[0], sdf_vals[0].unsqueeze(1)], dim=1)
            optimizer.zero_grad()
            
            num_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False
            sdf_data = sdf_data.cuda()

            # Debug: Print SDF range before clamping.
            if args.debug:
                print(f"Epoch {epoch+1}: Raw SDF range: min = {sdf_data[:,3].min().item():.4f}, max = {sdf_data[:,3].max().item():.4f}")

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)
                # Debug: Print SDF range after clamping.
                if args.debug:
                    print(f"Epoch {epoch+1}: Clamped SDF range: min = {sdf_gt.min().item():.4f}, max = {sdf_gt.max().item():.4f}")

            xyz_chunks = torch.chunk(xyz, batch_split)
            sdf_gt_chunks = torch.chunk(sdf_gt, batch_split)
            indices = torch.full((num_samples,), shape_idx.item(), dtype=torch.long).cuda()
            indices_chunks = torch.chunk(indices, batch_split)
            
            # with torch.amp.autocast("cuda"):
                # fp16_preds=[]
                # for i in range(batch_split):
                #     batch_vecs = lat_vecs(indices_chunks[i])
                #     input_chunk = torch.cat([batch_vecs, xyz_chunks[i]], dim=1)
                #     fp16_preds.append(decoder(input_chunk))
                    
            batch_loss = 0.0
                # batch_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)

            for i in range(batch_split):
                batch_vecs = lat_vecs(indices_chunks[i])
                input_chunk = torch.cat([batch_vecs, xyz_chunks[i]], dim=1)
            # for i, pred_sdf16 in enumerate(fp16_preds):
                pred_sdf = decoder(input_chunk)
                # pred_sdf = pred_sdf16.float()
                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                chunk_loss = loss_fn(pred_sdf, sdf_gt_chunks[i].cuda()) / num_samples
                if args.do_code_regularization:
                    l2_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (args.code_reg_lambda * min(1, epoch / 100) * l2_loss) / num_samples
                    chunk_loss += reg_loss.cuda()
                        # chunk_loss += reg_loss
                chunk_loss.backward()
                batch_loss += chunk_loss.item()
                    # batch_loss += chunk_loss

            # AMP: scale, backward, step, update
            # scaler.scale(batch_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if lat_vecs.weight.grad is not None:
                latent_grad_norm = lat_vecs.weight.grad.norm().item()
                if args.debug:
                    print(f"Epoch {epoch+1} latent grad norm: {latent_grad_norm:.6f}")

            optimizer.step()
            # Log iteration-level loss
            iteration_loss_log.append(batch_loss)
            epoch_loss_values.append(batch_loss)
            # iteration_loss_log.append(batch_loss.item())
            # epoch_loss_values.append(batch_loss.item())

            epoch_loss_sum += batch_loss
            epoch_num_batches += 1

        # Average loss for this epoch
        epoch_loss = epoch_loss_sum / epoch_num_batches
        epoch_loss_log.append(epoch_loss)
        elapsed = time.time() - start_time
        # timing_log.append(elapsed)
        mean_latent_norm = torch.mean(torch.norm(lat_vecs.weight.data, dim=1)).item()
        print(f"[Epoch {epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s | Mean latent norm: {mean_latent_norm:.6f}")
        
        # Debug: Every 10 epochs, run a latent sensitivity test on the first sample.
        if args.debug and (epoch % 10 == 0):
            test_latent_sensitivity(decoder, lat_vecs(torch.tensor([0]).cuda()), xyz_chunks[0])
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "decoder_state": decoder.state_dict(),
                "latents_state": lat_vecs.state_dict(),
            }, "auto_decoder.pth")
            print(f"New best model saved at epoch {epoch+1} with loss {epoch_loss:.6f}")
    
    if not args.visualize_only:
        plot_loss_curve(iteration_loss_log, epoch_loss_log, window=10, save_path="training_loss_curve.png")

    # Debug: Plot PCA of latents.
    if args.debug:
        plot_latent_distribution(lat_vecs, save_path="latent_distribution.png")

    # if args.segmentation_file:
    #     seg_mask = load_nifti_as_tensor(args.segmentation_file)
    #     latent_code = lat_vecs(torch.tensor([0]).cuda())
    #     visualize_sdf_slice(decoder, latent_code, seg_mask, slice_idx=None,
    #                          device="cuda", save_path="sdf_slice.png")
    #     print("Visualization saved as sdf_slice.png")

if __name__ == "__main__":
    main()
