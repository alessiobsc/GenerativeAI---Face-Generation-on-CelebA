import torch
import torch.nn as nn
import logging


class CVAE(nn.Module):
    """
    Advanced Conditional Variational Autoencoder.
    Combines the conditional logic (supervised) with robust architectural
    features like BatchNorm, Sequential blocks, and utility generation methods.
    Assumes input images are resized to 64x64.
    """

    def __init__(self, input_channel=3, latent_dim=128, num_cond=3):
        super(CVAE, self).__init__()

        # Initialize dimensions
        self.input_channel = input_channel
        self.latent_dim = latent_dim
        self.num_cond = num_cond

        logging.info(f"Initializing Advanced CVAE: latent_dim={latent_dim}, num_cond={num_cond}")

        # Define the encoder layers using Sequential
        # Input channel is (image_channels + condition_channels)
        self.encoder = nn.Sequential(
            # Input: (3 + num_cond) x 64 x 64
            nn.Conv2d(self.input_channel + self.num_cond, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            # Output: 256 x 4 x 4
        )

        # Define the linear layers for latent distribution parameters
        self.fc_mu = nn.Linear(256 * 4 * 4, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, self.latent_dim)

        # Define the initial linear layer for decoding (Latent + Condition)
        self.fc_decode = nn.Linear(self.latent_dim + self.num_cond, 256 * 4 * 4)

        # Define the decoder layers using Sequential
        self.decoder = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 32 x 32 x 32
            nn.ConvTranspose2d(32, self.input_channel, kernel_size=4, stride=2, padding=1),
            # Output: 3 x 64 x 64

            # Using Tanh because ground truth CelebA images are normalized to [-1, 1]
            nn.Tanh()
        )

    def encode(self, x, c):
        """Encodes the input image and condition into latent parameters."""
        batch_size = x.size(0)

        # Expand condition to match spatial dimensions of the image
        c_expanded = c.view(batch_size, self.num_cond, 1, 1).expand(-1, -1, x.size(2), x.size(3))

        # Concatenate image and expanded conditions along the channel dimension
        x_cond = torch.cat([x, c_expanded], dim=1)

        # Forward pass through the sequential encoder
        h = self.encoder(x_cond)
        h = h.view(batch_size, -1)  # Flatten

        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick to allow backpropagation."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """Decodes the combined latent vector and condition into an image."""
        batch_size = z.size(0)

        # Concatenate latent vector and condition attributes
        z_cond = torch.cat([z, c], dim=1)

        # Pass through linear layer and reshape for transposed convolutions
        h = self.fc_decode(z_cond)
        h = h.view(batch_size, 256, 4, 4)

        # Forward pass through the sequential decoder
        return self.decoder(h)

    def forward(self, x, c):
        """Standard forward pass during training."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

    # --- ADDED UTILITY METHODS ---

    def init_weights(self):
        """Initializes the weights of the model using Xavier Uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reconstruct(self, x, c):
        """Reconstructs the input image using its specific conditions."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c)

    def generate(self, c, device="cuda"):
        """
        Generates completely random new faces based on the provided conditions.
        Useful for inference and testing.
        """
        batch_size = c.size(0)
        # Generate pure noise
        z = torch.randn((batch_size, self.latent_dim)).to(device)
        # Decode noise along with the specific conditions
        return self.decode(z, c)