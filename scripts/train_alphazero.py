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



import concurrent.futures
import multiprocessing as mp

def _worker_play_game(args):
    state_dict, config, game_idx = args
    # Force workers onto CPU to prevent M4 Pro Metal (MPS) contention locks
    device = torch.device("cpu")
    model = WallzNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    env = WallzEnv()
    mcts = MCTS(model, num_simulations=config['mcts_simulations'])
    game_history = []
    seen_states = {}

    def state_key(e):
        return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())

    seen_states[state_key(env)] = 1
    terminal = False
    reward = 0.0
    step = 0

    while not terminal and step < config['max_steps']:
        temp = 1.0 if step < config['temp_moves'] else 0.0
        action_probs = mcts.get_action_prob(env, temperature=temp)
        game_history.append((env.get_observation(), action_probs, env.current_player))

        legal_mask = env.get_legal_action_mask()
        legal_actions = np.flatnonzero(legal_mask)
        probs = np.zeros(209)
        probs[legal_actions] = action_probs[legal_actions]
        
        total_prob = probs.sum()
        if total_prob <= 0:
            probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            probs /= total_prob

        if temp == 0:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        _, reward, terminal, _ = env.step(action)
        step += 1
        key = state_key(env)
        seen_states[key] = seen_states.get(key, 0) + 1

        if not terminal and seen_states[key] >= config['rep_limit']:
            break

    winner = None
    if terminal:
        winner = 1 if (reward == 1.0 and env.current_player == 2) else 2

    processed = []
    for obs, p, player in game_history:
        z = 0.0 if winner is None else (1.0 if player == winner else -1.0)
        processed.append((obs, p, z))

    return processed, terminal, step

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
        self.min_terminal_games = env_int("AZ_MIN_TERMINAL_GAMES", 1)
        self.max_steps_per_game = env_int("AZ_MAX_STEPS_PER_GAME", 80)
        self.temperature_moves = env_int("AZ_TEMPERATURE_MOVES", self.max_steps_per_game)
        self.repetition_limit = env_int("AZ_REPETITION_LIMIT", 3)
        self.avoid_repeats = env_flag("AZ_AVOID_REPEATS", True)
        self.timeout_policy = os.getenv("AZ_TIMEOUT_POLICY", "draw").strip().lower()
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
            f"save_every={self.save_every}, min_terminal_games={self.min_terminal_games}, "
            f"max_steps_per_game={self.max_steps_per_game}, temperature_moves={self.temperature_moves}, "
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
        """Infer epoch number from standard and named checkpoint files."""
        patterns = (
            r"alphazero_epoch_(\d+)(?:_v\d+)?\.pt",
            r"alphazero_interrupt_epoch_(\d+)(?:_v\d+)?\.pt",
            r".*epoch_(\d+)(?:_v\d+)?\.pt",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, path.name)
            if match:
                return int(match.group(1))
        return None

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

    def _unique_checkpoint_path(self, path: Path) -> Path:
        """Never overwrite existing epoch checkpoints; create _v2, _v3, ... instead."""
        if not path.exists():
            return path

        version = 2
        while True:
            candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
            if not candidate.exists():
                return candidate
            version += 1

    def save_checkpoint(self, epoch: int, interrupted: bool = False):
        if interrupted:
            target_path = self.checkpoint_dir / f"alphazero_interrupt_epoch_{epoch}.pt"
        else:
            target_path = self.checkpoint_dir / f"alphazero_epoch_{epoch}.pt"

        path = self._unique_checkpoint_path(target_path)
        if path != target_path:
            tqdm.write(f"⚠️ {target_path} already exists; saving without overwrite to {path}")

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
        """Pick an action from MCTS policy, escaping repeated states with legal-action fallback."""
        mcts_probs = np.asarray(action_probs, dtype=np.float64)
        legal_mask = env.get_legal_action_mask()
        legal_actions = np.flatnonzero(legal_mask)
        if len(legal_actions) == 0:
            raise RuntimeError("Environment returned no legal actions.")

        probs = np.zeros_like(mcts_probs)
        probs[legal_actions] = mcts_probs[legal_actions]
        if probs.sum() <= 0:
            probs[legal_actions] = 1.0 / len(legal_actions)

        filtered = probs.copy()
        avoided_repeat = False
        escaped_forced_repeat = False

        if self.avoid_repeats:
            non_repeat_actions = [
                int(action)
                for action in legal_actions
                if not self._action_repeats_state(env, int(action), seen_states)
            ]
            if non_repeat_actions:
                repeat_filter = np.zeros_like(filtered, dtype=bool)
                repeat_filter[non_repeat_actions] = True
                filtered[~repeat_filter] = 0.0
                avoided_repeat = len(non_repeat_actions) < len(legal_actions)

                # If MCTS put all probability mass on repeating actions, escape using all legal non-repeat moves.
                if filtered.sum() <= 0:
                    filtered[non_repeat_actions] = 1.0 / len(non_repeat_actions)
                    escaped_forced_repeat = True

        total = filtered.sum()
        if total <= 0:
            # Last-resort fallback: legal uniform. This should be rare, but keeps self-play alive.
            filtered = np.zeros_like(mcts_probs)
            filtered[legal_actions] = 1.0 / len(legal_actions)
            total = filtered.sum()

        filtered = filtered / total

        if temperature == 0:
            return int(np.argmax(filtered)), avoided_repeat, escaped_forced_repeat

        return int(np.random.choice(len(filtered), p=filtered)), avoided_repeat, escaped_forced_repeat

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
        self.model.eval()
        # Extract model state so it can be shipped safely to isolated processes
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        config = {
            'mcts_simulations': self.mcts_simulations,
            'max_steps': self.max_steps_per_game,
            'temp_moves': self.temperature_moves,
            'rep_limit': self.repetition_limit
        }
        args_list = [(state_dict, config, i) for i in range(self.games_per_epoch)]

        # Use up to 10 CPU cores on the M4 Pro to leave resources for system/MPS
        num_workers = min(os.cpu_count() or 4, 10) 
        completed = 0
        adjudicated = 0
        total_steps = 0

        print(f"\n  Spawning {num_workers} parallel workers for {self.games_per_epoch} self-play games...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_play_game, args) for args in args_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parallel Self-Play"):
                processed_history, terminal, steps = future.result()
                self.replay_buffer.extend(processed_history)
                total_steps += steps
                if terminal:
                    completed += 1
                else:
                    adjudicated += 1

        counted = completed + adjudicated
        return {
            "completed_games": completed,
            "adjudicated_games": adjudicated,
            "skipped_games": 0,
            "repeat_avoids": 0,
            "repeat_escapes": 0,
            "avg_steps": total_steps / counted if counted else 0,
            "replay_buffer": len(self.replay_buffer),
        }

    def train_network(self, terminal_games_this_epoch: int):
        """Trains the Neural Network only when new real terminal games were generated this epoch."""
        if terminal_games_this_epoch < self.min_terminal_games:
            message = (
                f"Terminal games this epoch too low: "
                f"{terminal_games_this_epoch}/{self.min_terminal_games}. Skipping network update."
            )
            if self.show_progress:
                tqdm.write(message)
            else:
                print(message)
            return None

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
        
        # Train for a few gradient steps proportional to the new data
        training_steps = max(10, len(self.replay_buffer) // self.batch_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(training_steps):
            batch = random.sample(self.replay_buffer, self.batch_size)
            state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
            prob_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
            value_batch = torch.FloatTensor(np.array([x[2] for x in batch]).astype(np.float32)).unsqueeze(1).to(self.device)

            logits, values = self.model(state_batch)
            
            policy_loss = -torch.sum(prob_batch * F.log_softmax(logits, dim=1), dim=1).mean()
            value_loss = F.mse_loss(values, value_batch)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        losses = {
            "policy_loss": total_policy_loss / training_steps,
            "value_loss": total_value_loss / training_steps,
            "total_loss": (total_policy_loss + total_value_loss) / training_steps,
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
            losses = self.train_network(stats["completed_games"])

            if self.show_progress:
                postfix = {
                    "epoch": f"{epoch}/{final_epoch}",
                    "games": stats["completed_games"],
                    "adj": stats["adjudicated_games"],
                    "skip": stats["skipped_games"],
                    "avg_steps": f"{stats['avg_steps']:.1f}",
                    "replay": stats["replay_buffer"],
                    "avoided": stats["repeat_avoids"],
                    "escapes": stats["repeat_escapes"],
                }
                if losses is not None:
                    postfix["loss"] = f"{losses['total_loss']:.3f}"
                else:
                    postfix["loss"] = "skipped"
                epoch_iter.set_postfix(postfix, refresh=True)

            if losses is not None and epoch % self.save_every == 0:
                self.save_checkpoint(epoch)
            elif losses is None:
                tqdm.write(f"⏭️ Not saving epoch {epoch}: model was not updated.")


if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    trainer = AlphaZeroTrainer()
    try:
        trainer.learn()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving interrupt checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, interrupted=True)
        raise
