import os
import torch
import logging
import config
from modules.conditional_vae import CVAE
from modules.config_setup import set_seed, setup_device
from modules.dataset import get_dataloaders
from modules.logging_setup import setup_logger
from modules.trainer import train_model

def main():
    """
    Main function to train the CVAE model.
    """
    # Initialize basic settings from config
    set_seed(config.SEED)
    device = setup_device()
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

    logging.info("Starting training script execution.")
    logging.info(f"Using device: {device}")
    logging.info(f"Hyperparameters -> LATENT_DIM: {config.LATENT_DIM}, NUM_COND: {config.NUM_COND}, BATCH_SIZE: {config.BATCH_SIZE}, EPOCHS: {config.EPOCHS}, LR: {config.LR}")

    # Check if we are running on the university HPC cluster, gracefully degrading to local path if not
    data_path = config.HPC_DATA_DIR
    if not os.path.exists(config.HPC_DATA_DIR):
        logging.warning(f"HPC data directory '{config.HPC_DATA_DIR}' not found.")
        raise FileNotFoundError("Dataset not found")

    logging.info(f"Resolved data path: {data_path}")

    try:
        # 1. Mount Data into Loaders
        logging.info("Initializing dataloaders...")
        train_loader, _ = get_dataloaders(data_dir=data_path, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

        # 2. Instantiate Model
        logging.info("Instantiating CVAE model...")
        cvae_model = CVAE(latent_dim=config.LATENT_DIM, num_cond=config.NUM_COND)
        logging.info(f"Model instantiated with {sum(p.numel() for p in cvae_model.parameters() if p.requires_grad)} trainable parameters.")

        # 3. Train Model
        logging.info("Handing over to the trainer...")
        trained_model = train_model(cvae_model, train_loader, device, epochs=config.EPOCHS, lr=config.LR)

        # 4. Save Model Weights
        logging.info("Saving model weights...")
        torch.save(trained_model.state_dict(), config.MODEL_WEIGHTS_PATH)
        logging.info(f"Model weights saved successfully to '{config.MODEL_WEIGHTS_PATH}'")

    except Exception as e:
        logging.error(f"Execution failed during training: {e}")
        raise

if __name__ == "__main__":
    setup_logger(log_dir="logs")
    main()