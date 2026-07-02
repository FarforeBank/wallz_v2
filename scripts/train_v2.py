import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from wallz_v2.env.gym_env import WallzGymEnv

class WallzFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.linear = nn.Sequential(
            nn.Linear(128 * 9 * 9, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def make_env():
    def mask_fn(env):
        return env.action_masks()
    
    env = WallzGymEnv()
    return ActionMasker(env, mask_fn)

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS (Apple Silicon) Available: {torch.backends.mps.is_available()}")
    
    print("\nInitializing Multiprocessing Vector Environment...")
    vec_env = make_vec_env(make_env, n_envs=15, vec_env_cls=SubprocVecEnv)

    save_path = ROOT_DIR / "wallz_v2" / "checkpoints" / "ppo_v2_model"
    model_file = Path(str(save_path) + ".zip")

    if model_file.exists():
        print(f"\n♻️ Found existing model at {model_file}!")
        print("Resuming training from where you left off...")
        model = MaskablePPO.load(model_file, env=vec_env, device="auto")
    else:
        print("\n✨ No existing model found. Creating a brand new one...")
        policy_kwargs = dict(
            features_extractor_class=WallzFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[128, 128]
        )
        model = MaskablePPO(
            "CnnPolicy", 
            vec_env, 
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="auto",
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
        )

    print("\n🚀 Starting Self-Play Training! (Press Ctrl+C to save and exit)...")
    try:
        # 🔥 CHANGED: Added progress_bar=True here!
        model.learn(total_timesteps=500_000, reset_num_timesteps=False, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    
    os.makedirs(save_path.parent, exist_ok=True)
    model.save(save_path)
    print(f"✅ Model saved safely to {save_path}.zip")

if __name__ == "__main__":
    main()
