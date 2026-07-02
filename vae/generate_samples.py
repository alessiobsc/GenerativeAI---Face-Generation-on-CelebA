import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from modules.conditional_vae import CVAE


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Setup and Model Initialization
    os.makedirs("samples", exist_ok=True)
    model = CVAE(input_channel=3, latent_dim=128, num_cond=3).to(device)

    # 2. Load trained weights
    checkpoint_path = "weights/cvae_weights_epoch83.ckp"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logging.info("Model weights loaded successfully!")
    else:
        logging.error(f"Checkpoint not found at {checkpoint_path}.")
        return

    model.eval()

    # 3. Define the 8 specific condition combinations and their labels
    # Order: [Male, Smiling, Young]
    conditions = [
        (torch.tensor([[0.0, 0.0, 0.0]]), "Femmina\nNon sorridente\nNon giovane\nclass=0"),
        (torch.tensor([[0.0, 0.0, 1.0]]), "Femmina\nNon sorridente\nGiovane\nclass=1"),
        (torch.tensor([[0.0, 1.0, 0.0]]), "Femmina\nSorridente\nNon giovane\nclass=2"),
        (torch.tensor([[0.0, 1.0, 1.0]]), "Femmina\nSorridente\nGiovane\nclass=3"),
        (torch.tensor([[1.0, 0.0, 0.0]]), "Maschio\nNon sorridente\nNon giovane\nclass=4"),
        (torch.tensor([[1.0, 0.0, 1.0]]), "Maschio\nNon sorridente\nGiovane\nclass=5"),
        (torch.tensor([[1.0, 1.0, 0.0]]), "Maschio\nSorridente\nNon giovane\nclass=6"),
        (torch.tensor([[1.0, 1.0, 1.0]]), "Maschio\nSorridente\nGiovane\nclass=7")
    ]

    cols_per_class = 4
    total_cols = cols_per_class + 1  # 1 column for text + 4 columns for images

    # 4. Initialize Matplotlib Figure
    # Adjust figsize to match the aspect ratio of your reference image
    fig, axes = plt.subplots(nrows=8, ncols=total_cols, figsize=(12, 20))
    fig.suptitle("CVAE CelebA - epoch=83", fontsize=16, y=0.98)

    logging.info("Starting generation and plotting...")

    with torch.no_grad():
        for row_idx, (cond_tensor, label) in enumerate(conditions):

            # --- Column 0: Text Label ---
            ax_text = axes[row_idx, 0]
            ax_text.axis('off')
            # Center the text vertically and horizontally in its grid cell
            ax_text.text(0.5, 0.5, label, ha='center', va='center', fontsize=11)

            # --- Columns 1 to 4: Generated Images ---
            # Expand the condition to generate 'cols_per_class' images at once
            cond_batch = cond_tensor.expand(cols_per_class, -1).to(device)
            gen_imgs = model.generate(cond_batch, device=device)

            # Denormalize images from [-1, 1] (Tanh) back to [0, 1] for Matplotlib
            gen_imgs = (gen_imgs + 1) / 2.0

            # Move to CPU and convert to numpy array
            gen_imgs = gen_imgs.cpu().numpy()

            for col_idx in range(cols_per_class):
                ax_img = axes[row_idx, col_idx + 1]
                ax_img.axis('off')

                # PyTorch uses (Channel, Height, Width), Matplotlib expects (Height, Width, Channel)
                img = np.transpose(gen_imgs[col_idx], (1, 2, 0))

                # Plot the image
                ax_img.imshow(img)

    # 5. Layout Formatting and Saving
    # Remove spacing between image plots to create a tight grid
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, bottom=0.01, left=0.05, right=0.98)

    output_path = os.path.join("samples", "cvae_generation_grid.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    logging.info(f"Grid successfully generated and saved at: {output_path}")


if __name__ == "__main__":
    main()