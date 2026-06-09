import numpy as np
import torch
import matplotlib.pyplot as plt
from vae import ConvVAE

# Load
frames = np.load('cartpole_frames/frames.npy')
pole_angles = np.load('cartpole_frames/pole_angles.npy')
frames_t = torch.FloatTensor(frames).permute(0, 3, 1, 2)

# Load trained model with latent_dim = 2
vae2 = ConvVAE(latent_dim = 2)
vae2.load_state_dict(torch.load('vae_weights_latent2.pth'))
vae2.eval()

# Encode all frames
mus = []
with torch.no_grad():
    for i in range(0, len(frames_t), 256):
        batch = frames_t[i: i+256]
        mu, _ = vae2.encoder(batch)
        mus.append(mu)

mus = torch.cat(mus).numpy()

# Plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(mus[:, 0], mus[:, 1], c = pole_angles, cmap='RdBu', alpha = 0.4, s = 8)
plt.colorbar(sc, label='Pole angle (rad)')
plt.xlabel('mu[0]')
plt.ylabel('mu[1]')
plt.title('CartPole VAE latent space (dim = 2)')
plt.savefig('cartpole_latent.png', dpi = 150)