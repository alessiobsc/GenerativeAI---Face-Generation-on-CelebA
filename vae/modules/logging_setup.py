import logging
import os
from datetime import datetime


def setup_logger(log_dir="logs"):
    """
    Configures a dual-handler logger to output messages to both the console (SLURM)
    and a persistent log file with a timestamp.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate a unique filename based on the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Configure the base logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Format for the logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 1. Console Handler (outputs to slurm-<JobID>.out)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (outputs to logs/training_YYYYMMDD_HHMMSS.log)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f"Logger initialized. Saving logs to: {log_file}")