import logging

import torch
import matplotlib.pyplot as plt


def generate_samples(model, device, num_samples=8, output_filename="cvae_samples.png"):
    """
    Generates face samples demonstrating conditional manipulation and saves them to disk.
    Modified for HPC execution (headless environment).
    """
    logging.info(f"Generating {num_samples} samples per condition set...")
    model.eval()
    with torch.no_grad():
        # Create a batch of random latent vectors from standard normal distribution
        z = torch.randn(num_samples * 2, model.latent_dim).to(device)

        # Build conditions: [Male, Smiling, Young]
        # Top row: 8 samples with (1.0, 1.0, 1.0) -> Male, Smiling, Young
        cond_msy = torch.tensor([[1.0, 1.0, 1.0]]).repeat(num_samples, 1).to(device)
        # Bottom row: 8 samples with (0.0, 0.0, 0.0) -> Female, Not Smiling, Not Young
        cond_fnny = torch.tensor([[0.0, 0.0, 0.0]]).repeat(num_samples, 1).to(device)

        cond = torch.cat([cond_msy, cond_fnny], dim=0)

        # Generate images using the decoder
        samples = model.decode(z, cond).cpu()

        # Un-normalize images from [-1, 1] back to [0, 1] for matplotlib plotting
        samples = (samples + 1.0) / 2.0

        # Visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        for i in range(num_samples):
            # Display Male, Smiling, Young
            axes[0, i].imshow(samples[i].permute(1, 2, 0).numpy())
            axes[0, i].axis('off')
            axes[0, i].set_title("Male, Smile, Young" if i == 0 else "")

            # Display Female, Not Smile, Not Young
            axes[1, i].imshow(samples[num_samples + i].permute(1, 2, 0).numpy())
            axes[1, i].axis('off')
            axes[1, i].set_title("Female, No Smile, No Young" if i == 0 else "")

        plt.tight_layout()

        logging.info(f"Saving generated samples plot to {output_filename}")
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)  # Free up memory

        logging.info(f"Samples generated and saved successfully as '{output_filename}'")