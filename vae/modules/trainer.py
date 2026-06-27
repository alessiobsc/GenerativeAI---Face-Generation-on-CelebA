import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
import logging
from torchvision.utils import save_image
from modules.loss import loss_function
from modules.checkpoint_manager import CheckpointManager


def train_model(model, dataloader, device, epochs=100, lr=1e-3):
    """
    Handles the training loop with time-based checkpointing using the Professor's CheckpointManager,
    and visual monitoring via image saving.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- SETUP FOR SAVING IMAGES ---
    samples_dir = "samples"
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    # Extract a fixed batch of images for visual progress monitoring
    fixed_data, fixed_cond = next(iter(dataloader))
    fixed_data = fixed_data[:32].to(device)
    fixed_cond = fixed_cond[:32].to(device)

    save_image(fixed_data, f"{samples_dir}/00_real_images.png", normalize=True)
    logging.info(f"Saved real images batch in {samples_dir}/00_real_images.png")

    # --- SETUP CHECKPOINT MANAGER ---
    # Initialize the manager, keeping only the last 3 checkpoints to save disk space
    cpm = CheckpointManager(folder='checkpoints', prefix='cvae_', kept_checkpoints=3)
    start_epoch = 1

    # Attempt to load the last available checkpoint
    cp = cpm.load_last_checkpoint(map_location=device)
    if cp is not None:
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        # Resume from the next epoch
        start_epoch = cp['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")

    model.train()

    # --- CHECKPOINT TIMER SETUP ---
    last_save_time = time.time()
    save_interval = 5 * 60  # 5 minutes in seconds

    for epoch in range(start_epoch, epochs + 1):
        train_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (data, cond) in enumerate(pbar):
            data = data.to(device)
            cond = cond.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, cond)

            loss, recon, kld = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}",
                              "Recon": f"{recon.item():.4f}",
                              "KLD": f"{kld.item():.4f}"})

            # --- TIME-BASED CHECKPOINTING ---
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                # Structure the state dictionary as required by the CheckpointManager
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss.item(),
                }
                cpm.save_checkpoint(state)
                logging.info(f"Auto-save triggered: Checkpoint saved at epoch {epoch}")
                last_save_time = current_time

        avg_loss = train_loss / len(dataloader)
        logging.info(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")

        # --- VISUAL MONITORING AT EPOCH END ---
        model.eval()
        with torch.no_grad():
            recon_fixed, _, _ = model(fixed_data, fixed_cond)
            # Save the reconstructed images for this epoch
            save_image(recon_fixed.cpu(), f"{samples_dir}/recon_epoch_{epoch:02d}.png", normalize=True)
        model.train()

        # --- FINAL CHECKPOINT ---
    # Ensure the very last epoch is saved when the loop completes naturally
    final_state = {
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    cpm.save_checkpoint(final_state)
    logging.info("Training complete. Final checkpoint saved.")

    return model