import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim = 32):
        super().__init__()
        # Input: (B, 3, 64, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 4, stride = 2, padding = 1),
            # -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
            # -> (B, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
            # -> (B, 128, 8, 8)
            nn.ReLU(),
        )
        self.flatten_dim = 128 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_lv = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x) # (B, 128, 8, 8)
        h = h.reshape(h.size(0), -1) # flatten: (B, 8192)
        return self.fc_mu(h), self.fc_lv(h)
    
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1), # -> (B, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1), # -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size = 4, stride = 2, padding = 1), # -> (B, 3, 64, 64)
            nn.Sigmoid(), # pixel values in [0, 1]
        )
    
    def forward(self, z):
        h = self.fc(z)
        h = h.reshape(h.size(0), 128, 8, 8)
        return self.deconv(h)

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps

class ConvVAE(nn.Module):
    def __init__(self, latent_dim = 32):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var