# CartPole On Latent Space
This project shows how the CartPole environment appears in latent space.

## The latent space plot
![Latent space plot](cartpole_latent.png)

## What I built

- 'Convolutional VAE':  3 convolution layer network
- 'frame collections': run cartpole environments randomly and save the frames
- use Adam to train

## What the latent space plot shows
Color represents the pole angle and as closer to blue, the angle gets bigger. These plots look smooth and structured. This shows that the pole angle is learnable by using this latent space.

## Result

| Metric              | Value        |
|---------------------|--------------|
| Minimum Loss        | 211.68       |
| Training epoch.     | 20           |
| Training time       | ~10 min (CPU)|

![Reconstruction comparison](cartpole_reconstructions.png)
This shows that this vae can reconstruct the frames correctly.

## How to run

```bash
pip install -r requirements.txt
python collect_frames.py  # collect frames for training
python train.py           # train VAE network, save the weights
python grid.py            # compare Original vs VAE reconstruction
python latent_scatter.py. # plot the lattent space plot
```

## Project structure

```
.
+-- train.py   # VAE training loop
+-- vae.py     # define vae network
+-- collect_frames.py # collect frames
+-- grid.py # visualize vae reconstruction 
+-- latent_scatter.py # plot the latent space
+-- requirements.txt
+-- cartpole_frame/
+-- cartpole_latent.png
+-- cartpole_reconstructions.png
+-- vae_weights_latent2.pth # weights when latent_dim = 2
+-- vae_weights_latent32.pth # weights when latent_dim = 32
+-- README.md
```

## Connection to Phase 3
This makes the world model be able to learn the space. 