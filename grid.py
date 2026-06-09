import numpy as np
import torch
from vae import ConvVAE
import matplotlib.pyplot as plt

# Load frames
frames = np.load('cartpole_frames/frames.npy') # (N, 64, 64, 3)

frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) # (N, 3, 64, 64)

idx = torch.randperm(len(frames))

sample = frames[idx]

vae = ConvVAE(latent_dim = 32)
vae.load_state_dict(torch.load('vae_weights_latent32.pth'))
vae.eval()

with torch.no_grad():
    x_hat, _, _ = vae(sample)

fig, axes = plt.subplots(2, 8, figsize = (16, 4))

for i in range(8):
    # Original: permute back to (H, W, C) for imshow
    axes[0, i].imshow(sample[i].permute(1, 2, 0).numpy())
    axes[0, i].axis('off')
    axes[1, i].imshow(x_hat[i].permute(1, 2, 0).numpy())
    axes[1, i].axis('off')

axes[0, 0].set_title('Original', loc = 'left')
axes[1, 0].set_title('Reconstructed', loc = 'left')
plt.tight_layout()
plt.savefig('cartpole_reconstructions.png', dpi = 150)
plt.show()