import torch
import torch.nn.functional as F
import logging

def loss_function(recon_x, x, mu, logvar):
    """Computes the VAE loss combining Reconstruction Error and KL Divergence."""
    # Reconstruction loss using MSE (suitable for values in [-1, 1] mapped by tanh)
    recon = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL Divergence analytical form: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_loss = recon + kld
    
    # We don't log every single loss computation here to avoid log spam, 
    # but we can assert for NaNs which is very useful for debugging VAEs
    if torch.isnan(total_loss):
        logging.error("NaN detected in loss computation! recon: {}, kld: {}, mu_mean: {}, logvar_mean: {}".format(
            recon.item(), kld.item(), mu.mean().item(), logvar.mean().item()
        ))
        
    return total_loss, recon, kld