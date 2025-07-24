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
from geometric_model.utils import load_nifti_as_tensor, visualize_sdf_slice, plot_loss_curve, test_latent_sensitivity, plot_latent_distribution, collate_shapes

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
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training.")

    args = parser.parse_args()
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    mapping_json = os.path.join(args.npz_folder, "mapping.json")

    if torch.cuda.is_available():
        print("CUDA available - using GPU.")
        args.device = torch.device("cuda")
    else:
        print("CUDA not available - using CPU.")
        args.device = torch.device("cpu")

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
        ).to(args.device)
        decoder.load_state_dict(checkpoint["decoder_state"])
        decoder.eval()
        n_shapes = len(checkpoint["latents_state"]["weight"])
        lat_vecs = torch.nn.Embedding(n_shapes, args.latent_size).to(args.device)
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_shapes)
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
    ).to(args.device)

    # Create latent embedding with improved initialization.
    lat_vecs = nn.Embedding(n_shapes, args.latent_size)
    init_std = 1.0 / (args.latent_size ** 0.5)
    torch.nn.init.normal_(lat_vecs.weight, 0.0, init_std)
    lat_vecs = lat_vecs.to(args.device)

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
            shape_idxs, coords, sdf_vals = dataset[i]
            print(f"Shape {dataset.shape_names[i]}: SDF min = {sdf_vals.min().item():.4f}, max = {sdf_vals.max().item():.4f}")

    # Speedup with amp:
    scaler = torch.amp.GradScaler()
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        decoder.train()
        epoch_loss_sum = 0.0
        epoch_num_batches = 0
        epoch_loss_values = []
        
        for shape_idxs, coords_list, sdf_list in dataloader:
            # Concatenate all points into one big tensor
            coords = torch.cat(coords_list, dim=0).cuda()
            sdf_gt = torch.cat([s.unsqueeze(1) for s in sdf_list], dim=0).cuda()
            
            # Build a long index array so each point picks the right latent
            indices = torch.cat([
                torch.full((c.shape[0],), idx, dtype=torch.long)
                for idx, c in zip(shape_idxs, coords_list)
            ], dim=0).cuda()
            N_total = coords.shape[0]

            # Split coords & sdf_gt into sub-batches to avoid OOM (exactly what batch_split did)
            coords_chunks = coords.chunk(batch_split)
            sdf_chunks = sdf_gt.chunk(batch_split)
            idx_chunks = indices.chunk(batch_split)

            optimizer.zero_grad()
            for xc, sf, idxc in zip(coords_chunks, sdf_chunks, idx_chunks):
                zc = lat_vecs(idxc)
                inp = torch.cat([zc, xc], dim=1)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = decoder(inp)
                    if enforce_minmax:
                        pred = torch.clamp(pred, minT, maxT)
                    loss = loss_fn(pred, sf) / N_total

                if args.do_code_regularization:
                    loss = loss + (args.code_reg_lambda * min(1, epoch/100)
                                * zc.norm(dim=1).sum() / N_total)

                scaler.scale(loss).backward()
                
            if lat_vecs.weight.grad is not None:
                latent_grad_norm = lat_vecs.weight.grad.norm().item()
                if args.debug:
                    print(f"Epoch {epoch+1} latent grad norm: {latent_grad_norm:.6f}")

            scaler.step(optimizer)
            scaler.update()
            # Log iteration-level loss
            iteration_loss_log.append(loss)
            epoch_loss_values.append(loss)

            epoch_loss_sum += loss
            epoch_num_batches += 1

        # Average loss for this epoch
        epoch_loss = epoch_loss_sum / epoch_num_batches
        epoch_loss_log.append(epoch_loss)
        elapsed = time.time() - start_time
        mean_latent_norm = torch.mean(torch.norm(lat_vecs.weight.data, dim=1)).item()
        print(f"[Epoch {epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s | Mean latent norm: {mean_latent_norm:.6f}")
        
        # Debug: Every 10 epochs, run a latent sensitivity test on the first sample.
        if args.debug and (epoch % 10 == 0):
            test_latent_sensitivity(decoder, lat_vecs(torch.tensor([0]).cuda()), coords_chunks[0])
            
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

if __name__ == "__main__":
    main()
