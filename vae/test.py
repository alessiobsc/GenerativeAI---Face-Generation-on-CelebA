import torch
import os
import logging
from torchvision.utils import save_image
import config
from modules.checkpoint_manager import CheckpointManager
from modules.conditional_vae import CVAE
from modules.config_setup import set_seed, setup_device
from modules.dataset import get_dataloaders
from modules.logging_setup import setup_logger
from modules.loss import loss_function


def test_model(model, dataloader, device):
    """
    Evaluates the model on the unseen test dataset.
    Returns the average loss components.
    """
    model.eval()
    test_loss = 0.0
    test_recon = 0.0
    test_kld = 0.0

    with torch.no_grad():
        for data, cond in dataloader:
            data = data.to(device)
            cond = cond.to(device)

            recon_batch, mu, logvar = model(data, cond)
            loss, recon, kld = loss_function(recon_batch, data, mu, logvar)

            test_loss += loss.item()
            test_recon += recon.item()
            test_kld += kld.item()

    num_batches = len(dataloader)
    avg_loss = test_loss / num_batches
    avg_recon = test_recon / num_batches
    avg_kld = test_kld / num_batches

    logging.info(f"Test Results - Total Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f}")
    return avg_loss


def generate_new_faces(model, dataloader, device, output_dir="samples", num_samples=32):
    """
    Generates completely new faces starting from pure random noise and test set conditions.
    """
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Get a batch of conditions from the test set (e.g., labels for hair color, gender, etc.)
    _, test_cond = next(iter(dataloader))
    test_cond = test_cond[:num_samples].to(device)

    # 2. Sample pure random noise from a standard normal distribution N(0, 1)
    # The latent dimension size must match your model's config (e.g., 128)
    latent_dim = model.latent_dim if hasattr(model, 'latent_dim') else config.LATENT_DIM
    z = torch.randn(num_samples, latent_dim).to(device)

    # 3. Decode the noise + conditions into new images
    with torch.no_grad():
        # Your decoder might take z and cond as input, depending on your architecture
        generated_images = model.decode(z, test_cond)

    # 4. Save the generated images
    output_path = os.path.join(output_dir, "00_FINAL_generated_faces.png")
    save_image(generated_images.cpu(), output_path, normalize=True)
    logging.info(f"{num_samples} brand new faces generated and saved in {output_path}")


def main():
    """
    Main execution flow for testing the CVAE.
    """
    setup_logger(log_dir="logs")
    set_seed(config.SEED)
    device = setup_device()

    logging.info("Starting Testing Phase...")

    # Load test dataloader
    _, test_loader = get_dataloaders(config.HPC_DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    # Initialize the model
    model = CVAE(latent_dim=config.LATENT_DIM, num_cond=config.NUM_COND).to(device)

    cpm = CheckpointManager(folder='checkpoints', prefix='cvae_')
    cp = cpm.load_last_checkpoint(map_location=device)

    if cp is not None:
        model.load_state_dict(cp['model'])
        logging.info(f"Loaded weights successfully (from Epoch {cp['epoch']})")
    else:
        logging.error("No checkpoint files found in the 'checkpoints' folder! Cannot test.")
        return

    # Run the evaluations
    test_model(model, test_loader, device)
    generate_new_faces(model, test_loader, device)


if __name__ == "__main__":
    main()