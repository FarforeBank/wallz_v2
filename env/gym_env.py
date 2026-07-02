import gymnasium as gym
import numpy as np
from gymnasium import spaces
from wallz_v2.env.wallz_env import WallzEnv

class WallzGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = WallzEnv()
        self.action_space = spaces.Discrete(209)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 9, 9), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, mask = self.env.reset()
        return obs, {"action_mask": mask}

    def step(self, action):
        obs, reward, terminal, mask = self.env.step(action)
        return obs, reward, terminal, False, {"action_mask": mask}
        
    def action_masks(self):
        return self.env.get_legal_action_mask()
