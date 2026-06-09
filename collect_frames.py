import gymnasium as gym
import numpy as np
from PIL import Image
import os

def collect_frames(n_frames=10_000, save_dir = 'cartpole_frames', img_size=(64,64)):
    os.makedirs(save_dir, exist_ok = True)
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    frames = []
    pole_angles = []
    state, _ = env.reset()
    count = 0

    while count < n_frames:
        # Random action (we just want diverse frames)
        action = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(action)

        # Render and resize
        frame = env.render()    # (H, W, 3)

        img = Image.fromarray(frame).resize(img_size)  # resize to 64 x 64

        frames.append(np.array(img))
        pole_angles.append(next_state[2])

        if terminated or truncated:
            state, _ = env.reset()
        count += 1

    env.close()
    frames = np.array(frames, dtype=np.float32) / 255.0 # [0, 1]
    np.save(f'{save_dir}/frames.npy', frames)
    np.save(f'{save_dir}/pole_angles.npy', np.array(pole_angles))
    print(f"Saved {len(frames)} frames to {save_dir} / frames.npy")
    return frames

if __name__ == '__main__':
    collect_frames()