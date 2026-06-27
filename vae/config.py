import logging

# Data settings
HPC_DATA_DIR = '/home/pfoggia/GenerativeAI/CELEBA'

# Model settings
LATENT_DIM = 128
NUM_COND = 3
MODEL_WEIGHTS_PATH = "cvae_celeba.pth"

# Training settings
SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 1
EPOCHS = 100
LR = 1e-3

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
