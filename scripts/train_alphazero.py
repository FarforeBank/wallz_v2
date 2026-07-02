import os
import re
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


def env_int(name: str, default: int) -> int:
    """Read a positive integer from env, falling back to default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"⚠️ Ignoring invalid {name}={value!r}; using {default}")
        return default
    return parsed if parsed > 0 else default


class AlphaZeroTrainer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = WallzNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Fast diagnostic defaults. Override without editing code, for example:
        # AZ_EPOCHS=50 AZ_GAMES_PER_EPOCH=10 AZ_MCTS_SIMULATIONS=25 python scripts/train_alphazero.py
        self.epochs = env_int("AZ_EPOCHS", 10)
        self.games_per_epoch = env_int("AZ_GAMES_PER_EPOCH", 2)
        self.mcts_simulations = env_int("AZ_MCTS_SIMULATIONS", 5)
        self.batch_size = env_int("AZ_BATCH_SIZE", 64)
        self.save_every = env_int("AZ_SAVE_EVERY", 1)
        self.max_steps_per_game = env_int("AZ_MAX_STEPS_PER_GAME", 200)
        self.replay_buffer = deque(maxlen=10000)

        self.checkpoint_dir = ROOT_DIR / "wallz_v2" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = self._load_latest_checkpoint() + 1
        self.current_epoch = self.start_epoch - 1

        print(
            "Config -> "
            f"epochs={self.epochs}, games_per_epoch={self.games_per_epoch}, "
            f"mcts_simulations={self.mcts_simulations}, batch_size={self.batch_size}, "
            f"save_every={self.save_every}, max_steps_per_game={self.max_steps_per_game}"
        )

    def _checkpoint_epoch(self, path: Path):
        match = re.fullmatch(r"alphazero_epoch_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else None

    def _load_latest_checkpoint(self) -> int:
        checkpoints = []
        for path in self.checkpoint_dir.glob("alphazero_epoch_*.pt"):
            epoch = self._checkpoint_epoch(path)
            if epoch is not None:
                checkpoints.append((epoch, path))

        if not checkpoints:
            print("No AlphaZero checkpoint found. Starting from scratch.")
            return 0

        epoch, path = max(checkpoints, key=lambda item: item[0])
        print(f"♻️ Loading AlphaZero checkpoint: {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Resuming after epoch {epoch}.")
        return epoch

    def save_checkpoint(self, epoch: int, interrupted: bool = False):
        if interrupted:
            path = self.checkpoint_dir / f"alphazero_interrupt_epoch_{epoch}.pt"
        else:
            path = self.checkpoint_dir / f"alphazero_epoch_{epoch}.pt"

        torch.save(self.model.state_dict(), path)
        latest_path = self.checkpoint_dir / "alphazero_latest.pt"
        torch.save(self.model.state_dict(), latest_path)
        print(f"💾 Saved checkpoint to {path}")
        print(f"💾 Updated latest checkpoint at {latest_path}")

    def self_play(self):
        """Generates training data by having the network play against itself using MCTS."""
        print(f"\n🎮 Generating {self.games_per_epoch} self-play games...")
        self.model.eval()

        for game in range(self.games_per_epoch):
            env = WallzEnv()
            mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
            game_history = []

            terminal = False
            reward = 0.0
            step = 0

            while not terminal and step < self.max_steps_per_game:
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

            if not terminal:
                print(f"⚠️ Game {game + 1}/{self.games_per_epoch} hit max_steps={self.max_steps_per_game}; skipping it.")
                continue

            # Game over, assign final values to the history buffer
            winner = 1 if (reward == 1.0 and env.current_player == 2) else 2

            for obs, probs, player in game_history:
                # Value is +1 if this player won, -1 if they lost
                z = 1.0 if player == winner else -1.0
                self.replay_buffer.append((obs, probs, z))

            print(f"Game {game + 1}/{self.games_per_epoch} complete (Steps: {step}). Winner: P{winner}")

    def train_network(self):
        """Trains the Neural Network using the experiences gathered by MCTS."""
        if len(self.replay_buffer) < self.batch_size:
            print(f"Replay buffer too small: {len(self.replay_buffer)}/{self.batch_size}. Skipping network update.")
            return

        print("\n🧠 Training Neural Network...")
        self.model.train()

        batch = random.sample(self.replay_buffer, self.batch_size)
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
        final_epoch = self.start_epoch + self.epochs - 1
        for epoch in range(self.start_epoch, final_epoch + 1):
            self.current_epoch = epoch
            print(f"\n{'=' * 40}\n AlphaZero Epoch {epoch}/{final_epoch}\n{'=' * 40}")
            self.self_play()
            self.train_network()

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)


if __name__ == '__main__':
    trainer = AlphaZeroTrainer()
    try:
        trainer.learn()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving interrupt checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, interrupted=True)
        raise
