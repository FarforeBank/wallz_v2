import copy
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
PACKAGE_DIR = ROOT_DIR / "wallz_v2"
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


def env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean-ish flag from env."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class AlphaZeroTrainer:
    def __init__(self):
        # CPU is now the safe default for AlphaZero/MCTS. Override with AZ_DEVICE=auto, mps, cuda, or cpu.
        self.cpu_threads = env_int("AZ_TORCH_THREADS", 20)
        self.cpu_interop_threads = env_int("AZ_TORCH_INTEROP_THREADS", 1)
        torch.set_num_threads(self.cpu_threads)
        try:
            torch.set_num_interop_threads(self.cpu_interop_threads)
        except RuntimeError as exc:
            print(f"⚠️ Could not set interop threads after torch initialization: {exc}")

        self.device = self._select_device()
        print(f"Using device: {self.device}")
        print(f"Torch CPU threads: {torch.get_num_threads()}")
        print(f"Torch CPU interop threads: {torch.get_num_interop_threads()}")

        self.model = WallzNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Fast diagnostic defaults. Override without editing code, for example:
        # AZ_EPOCHS=50 AZ_GAMES_PER_EPOCH=10 AZ_MCTS_SIMULATIONS=25 python scripts/train_alphazero.py
        self.epochs = env_int("AZ_EPOCHS", 10)
        self.games_per_epoch = env_int("AZ_GAMES_PER_EPOCH", 2)
        self.mcts_simulations = env_int("AZ_MCTS_SIMULATIONS", 5)
        self.batch_size = env_int("AZ_BATCH_SIZE", 64)
        self.save_every = env_int("AZ_SAVE_EVERY", 1)
        self.max_steps_per_game = env_int("AZ_MAX_STEPS_PER_GAME", 80)
        self.repetition_limit = env_int("AZ_REPETITION_LIMIT", 3)
        self.avoid_repeats = env_flag("AZ_AVOID_REPEATS", True)
        self.timeout_policy = os.getenv("AZ_TIMEOUT_POLICY", "distance").strip().lower()
        if self.timeout_policy not in {"distance", "draw", "skip"}:
            print(f"⚠️ Unknown AZ_TIMEOUT_POLICY={self.timeout_policy!r}; using 'distance'.")
            self.timeout_policy = "distance"
        self.show_progress = env_flag("AZ_PROGRESS", True)
        self.replay_buffer = deque(maxlen=10000)

        self.checkpoint_dir = PACKAGE_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = self._load_latest_checkpoint() + 1
        self.current_epoch = self.start_epoch - 1

        print(
            "Config -> "
            f"epochs={self.epochs}, games_per_epoch={self.games_per_epoch}, "
            f"mcts_simulations={self.mcts_simulations}, batch_size={self.batch_size}, "
            f"save_every={self.save_every}, max_steps_per_game={self.max_steps_per_game}, "
            f"repetition_limit={self.repetition_limit}, avoid_repeats={self.avoid_repeats}, "
            f"timeout_policy={self.timeout_policy}, "
            f"device={self.device}, torch_threads={torch.get_num_threads()}, "
            f"progress={self.show_progress}"
        )

    def _select_device(self) -> torch.device:
        requested = os.getenv("AZ_DEVICE", "cpu").strip().lower()

        if requested == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            print("⚠️ AZ_DEVICE=mps requested, but MPS is unavailable. Falling back to CPU.")
            return torch.device("cpu")

        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("⚠️ AZ_DEVICE=cuda requested, but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")

        if requested != "cpu":
            print(f"⚠️ Unknown AZ_DEVICE={requested!r}; using CPU.")
        return torch.device("cpu")

    def _checkpoint_epoch(self, path: Path):
        match = re.fullmatch(r"alphazero_epoch_(\d+)\.pt", path.name)
        if match:
            return int(match.group(1))
        match = re.fullmatch(r"alphazero_interrupt_epoch_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else None

    def _resolve_checkpoint_override(self):
        requested = os.getenv("AZ_RESUME_FROM")
        if not requested:
            return None

        raw_path = Path(requested).expanduser()
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(PACKAGE_DIR / raw_path)
            candidates.append(ROOT_DIR / raw_path)

        for path in candidates:
            if path.exists():
                return path

        formatted = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"AZ_RESUME_FROM={requested!r} was not found. Tried: {formatted}")

    def _load_latest_checkpoint(self) -> int:
        override_path = self._resolve_checkpoint_override()
        if override_path is not None:
            epoch = self._checkpoint_epoch(override_path) or 0
            print(f"♻️ Loading AlphaZero checkpoint from AZ_RESUME_FROM: {override_path}")
            state_dict = torch.load(override_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Resuming after epoch {epoch}.")
            return epoch

        checkpoints = []
        for pattern in ("alphazero_epoch_*.pt", "alphazero_interrupt_epoch_*.pt"):
            for path in self.checkpoint_dir.glob(pattern):
                epoch = self._checkpoint_epoch(path)
                if epoch is not None:
                    checkpoints.append((epoch, path.stat().st_mtime, path))

        if not checkpoints:
            print("No AlphaZero checkpoint found. Starting from scratch.")
            return 0

        epoch, _, path = max(checkpoints, key=lambda item: (item[0], item[1]))
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
        tqdm.write(f"💾 Saved checkpoint to {path}")
        tqdm.write(f"💾 Updated latest checkpoint at {latest_path}")

    def _state_key(self, env: WallzEnv):
        """Compact repeat-detection key for training only; does not change environment rules."""
        return (
            env.p1_pos,
            env.p2_pos,
            env.current_player,
            env.walls_left[1],
            env.walls_left[2],
            env.h_walls.tobytes(),
            env.v_walls.tobytes(),
        )

    def _action_repeats_state(self, env: WallzEnv, action: int, seen_states: dict) -> bool:
        sim_env = copy.deepcopy(env)
        sim_env.step(int(action))
        return self._state_key(sim_env) in seen_states

    def _select_self_play_action(self, env: WallzEnv, action_probs, temperature: float, seen_states: dict):
        """Pick an action from MCTS policy, optionally filtering one-step repeated states."""
        probs = np.asarray(action_probs, dtype=np.float64)
        legal_actions = np.flatnonzero(probs > 0)
        if len(legal_actions) == 0:
            raise RuntimeError("MCTS returned no legal actions.")

        filtered = probs.copy()
        avoided_repeat = False

        if self.avoid_repeats:
            non_repeat_actions = [
                int(action)
                for action in legal_actions
                if not self._action_repeats_state(env, int(action), seen_states)
            ]
            if non_repeat_actions:
                mask = np.zeros_like(filtered, dtype=bool)
                mask[non_repeat_actions] = True
                filtered[~mask] = 0.0
                avoided_repeat = len(non_repeat_actions) < len(legal_actions)

        total = filtered.sum()
        if total <= 0:
            filtered = probs.copy()
            total = filtered.sum()

        if temperature == 0:
            return int(np.argmax(filtered)), avoided_repeat

        filtered = filtered / total
        return int(np.random.choice(len(filtered), p=filtered)), avoided_repeat

    def _adjudicate_non_terminal_game(self, env: WallzEnv):
        """Resolve training-only non-terminal games caused by max-step or repetition limits."""
        if self.timeout_policy == "skip":
            return "skip", None
        if self.timeout_policy == "draw":
            return "draw", None

        p1_distance = env._get_bfs_distance(env.p1_pos, 0)
        p2_distance = env._get_bfs_distance(env.p2_pos, 8)

        if p1_distance < p2_distance:
            return "distance", 1
        if p2_distance < p1_distance:
            return "distance", 2
        return "draw", None

    def _append_game_history(self, game_history, winner):
        for obs, probs, player in game_history:
            if winner is None:
                z = 0.0
            else:
                # Value is +1 if this player won, -1 if they lost
                z = 1.0 if player == winner else -1.0
            self.replay_buffer.append((obs, probs, z))

    def self_play(self):
        """Generates training data by having the network play against itself using MCTS."""
        self.model.eval()
        game_iter = range(self.games_per_epoch)
        if self.show_progress:
            game_iter = tqdm(
                game_iter,
                total=self.games_per_epoch,
                desc="Self-play games",
                unit="game",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            print(f"\n🎮 Generating {self.games_per_epoch} self-play games...")

        completed_games = 0
        adjudicated_games = 0
        skipped_games = 0
        repeat_avoids = 0
        total_steps = 0

        for game in game_iter:
            env = WallzEnv()
            mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
            game_history = []
            seen_states = {self._state_key(env): 1}

            terminal = False
            reward = 0.0
            step = 0
            stop_reason = None

            while not terminal and step < self.max_steps_per_game:
                # Use temperature=1.0 for the first 15 moves to encourage exploration, then 0 to play strict best moves
                temp = 1.0 if step < 15 else 0.0

                # MCTS thinking
                action_probs = mcts.get_action_prob(env, temperature=temp)

                # Store state and target policy (from MCTS)
                game_history.append((env.get_observation(), action_probs, env.current_player))

                action, avoided_repeat = self._select_self_play_action(env, action_probs, temp, seen_states)
                if avoided_repeat:
                    repeat_avoids += 1

                _, reward, terminal, _ = env.step(action)
                step += 1

                state_key = self._state_key(env)
                seen_states[state_key] = seen_states.get(state_key, 0) + 1
                if not terminal and seen_states[state_key] >= self.repetition_limit:
                    stop_reason = "repetition"
                    break

                if self.show_progress and step % 5 == 0:
                    game_iter.set_postfix(
                        game=game + 1,
                        step=step,
                        replay=len(self.replay_buffer),
                        avoided=repeat_avoids,
                        refresh=False,
                    )

            if not terminal and stop_reason is None:
                stop_reason = "max_steps"

            if terminal:
                # Game over, assign final values to the history buffer
                winner = 1 if (reward == 1.0 and env.current_player == 2) else 2
                self._append_game_history(game_history, winner)
                completed_games += 1
                total_steps += step
                status = f"winner=P{winner}"
            else:
                outcome, winner = self._adjudicate_non_terminal_game(env)
                if outcome == "skip":
                    skipped_games += 1
                    message = (
                        f"⚠️ Game {game + 1}/{self.games_per_epoch} stopped by {stop_reason} "
                        f"at step={step}; skipping it."
                    )
                    if self.show_progress:
                        tqdm.write(message)
                    else:
                        print(message)
                    continue

                self._append_game_history(game_history, winner)
                adjudicated_games += 1
                total_steps += step
                status = "draw" if winner is None else f"adjudicated=P{winner}"
                message = (
                    f"⚖️ Game {game + 1}/{self.games_per_epoch} stopped by {stop_reason} "
                    f"at step={step}; {status}."
                )
                if self.show_progress:
                    tqdm.write(message)
                else:
                    print(message)

            if self.show_progress:
                game_iter.set_postfix(
                    game=game + 1,
                    steps=step,
                    status=status,
                    replay=len(self.replay_buffer),
                    avoided=repeat_avoids,
                    refresh=True,
                )
            else:
                print(f"Game {game + 1}/{self.games_per_epoch} complete (Steps: {step}). {status}")

        counted_games = completed_games + adjudicated_games
        avg_steps = total_steps / counted_games if counted_games else 0.0
        return {
            "completed_games": completed_games,
            "adjudicated_games": adjudicated_games,
            "skipped_games": skipped_games,
            "repeat_avoids": repeat_avoids,
            "avg_steps": avg_steps,
            "replay_buffer": len(self.replay_buffer),
        }

    def train_network(self):
        """Trains the Neural Network using the experiences gathered by MCTS."""
        if len(self.replay_buffer) < self.batch_size:
            message = f"Replay buffer too small: {len(self.replay_buffer)}/{self.batch_size}. Skipping network update."
            if self.show_progress:
                tqdm.write(message)
            else:
                print(message)
            return None

        if self.show_progress:
            tqdm.write("🧠 Training Neural Network...")
        else:
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

        losses = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }
        message = (
            f"Loss -> Policy: {losses['policy_loss']:.4f} | "
            f"Value: {losses['value_loss']:.4f} | Total: {losses['total_loss']:.4f}"
        )
        if self.show_progress:
            tqdm.write(message)
        else:
            print(message)
        return losses

    def learn(self):
        final_epoch = self.start_epoch + self.epochs - 1
        epoch_iter = range(self.start_epoch, final_epoch + 1)
        if self.show_progress:
            epoch_iter = tqdm(
                epoch_iter,
                total=self.epochs,
                desc="AlphaZero epochs",
                unit="epoch",
                dynamic_ncols=True,
            )

        for epoch in epoch_iter:
            self.current_epoch = epoch
            if not self.show_progress:
                print(f"\n{'=' * 40}\n AlphaZero Epoch {epoch}/{final_epoch}\n{'=' * 40}")

            stats = self.self_play()
            losses = self.train_network()

            if self.show_progress:
                postfix = {
                    "epoch": f"{epoch}/{final_epoch}",
                    "games": stats["completed_games"],
                    "adj": stats["adjudicated_games"],
                    "avg_steps": f"{stats['avg_steps']:.1f}",
                    "replay": stats["replay_buffer"],
                    "avoided": stats["repeat_avoids"],
                }
                if losses is not None:
                    postfix["loss"] = f"{losses['total_loss']:.3f}"
                epoch_iter.set_postfix(postfix, refresh=True)

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
