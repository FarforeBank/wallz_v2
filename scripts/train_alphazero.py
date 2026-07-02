import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS

class AlphaZeroTrainer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = WallzNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        self.epochs = 50
        self.games_per_epoch = 10
        self.mcts_simulations = 25 # Keep low for Python speed. Increase for smarter AI.
        self.replay_buffer = deque(maxlen=10000)
        
    def self_play(self):
        """Generates training data by having the network play against itself using MCTS."""
        print(f"\n🎮 Generating {self.games_per_epoch} self-play games...")
        self.model.eval()
        
        for game in range(self.games_per_epoch):
            env = WallzEnv()
            mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
            game_history = []
            
            terminal = False
            step = 0
            
            while not terminal:
                # Use temperature=1.0 for the first 15 moves to encourage exploration, then 0 to play strict best moves
                temp = 1.0 if step < 15 else 0.0
                
                # MCTS thinking
                action_probs = mcts.get_action_prob(env, temperature=temp)
                
                # Store state and target policy (from MCTS)
                game_history.append((env.get_observation(), action_probs, env.current_player))
                
                # Sample action
                if temp == 0:
                    action = np.argmax(action_probs)
                else:
                    action = np.random.choice(len(action_probs), p=action_probs)
                
                _, reward, terminal, _ = env.step(action)
                step += 1
            
            # Game over, assign final values to the history buffer
            winner = 1 if (reward == 1.0 and env.current_player == 2) else 2
            
            for obs, probs, player in game_history:
                # Value is +1 if this player won, -1 if they lost
                z = 1.0 if player == winner else -1.0
                self.replay_buffer.append((obs, probs, z))
                
            print(f"Game {game+1}/{self.games_per_epoch} complete (Steps: {step}). Winner: P{winner}")

    def train_network(self):
        """Trains the Neural Network using the experiences gathered by MCTS."""
        if len(self.replay_buffer) < 256:
            return
            
        print("\n🧠 Training Neural Network...")
        self.model.train()
        
        batch = random.sample(self.replay_buffer, 256)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        prob_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
        value_batch = torch.FloatTensor(np.array([x[2] for x in batch]).astype(np.float32)).unsqueeze(1).to(self.device)
        
        # We don't mask actions here because MCTS prob_batch already has 0s for illegal moves
        logits, values = self.model(state_batch)
        
        # Policy Loss: Cross Entropy between NN logits and MCTS probabilities
        policy_loss = -torch.sum(prob_batch * F.log_softmax(logits, dim=1), dim=1).mean()
        
        # Value Loss: Mean Squared Error between NN prediction and actual game outcome
        value_loss = F.mse_loss(values, value_batch)
        
        total_loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        print(f"Loss -> Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f} | Total: {total_loss.item():.4f}")

    def learn(self):
        for epoch in range(1, self.epochs + 1):
            print(f"\n{'='*40}\n AlphaZero Epoch {epoch}/{self.epochs}\n{'='*40}")
            self.self_play()
            self.train_network()
            
            # Save checkpoint
            if epoch % 5 == 0:
                save_path = ROOT_DIR / "wallz_v2" / "checkpoints" / f"alphazero_epoch_{epoch}.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"💾 Saved checkpoint to {save_path}")

if __name__ == '__main__':
    trainer = AlphaZeroTrainer()
    trainer.learn()
