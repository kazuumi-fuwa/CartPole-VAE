import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from vae import ConvVAE

def vae_loss(x_hat, x, mu, log_var):
    recon = F.binary_cross_entropy(x_hat, x, reduction = 'sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl

def main():
    latent_dim = 2
    # Load frames
    frames = np.load('cartpole_frames/frames.npy') # (N, 64, 64, 3)

    frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) # (N, 3, 64, 64)

    dataset = TensorDataset(frames)
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = True)

    vae = ConvVAE(latent_dim = latent_dim)
    opt = optim.Adam(vae.parameters(), lr = 1e-3)
    N_EPOCHS = 20

    for epoch in range(N_EPOCHS):
        total_loss = 0
        for (x,) in dataloader:
            x_hat, mu, log_var = vae(x)
            loss = vae_loss(x_hat, x, mu, log_var)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(frames)
        print(f"Epoch {epoch + 1:2d} | Loss: {avg:.2f}")
    
    torch.save(vae.state_dict(), f'vae_weights_latent{latent_dim}.pth')
    print(f"Model saved to vae_weights_latent{latent_dim}.pth")
    
if __name__ == '__main__':
    main()