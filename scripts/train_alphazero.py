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
import concurrent.futures
import multiprocessing as mp

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / "wallz_v2"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.env.action_space import TOTAL_ACTIONS
from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS, invert_action_array, flip_action_array_horizontal, flip_obs_horizontal, get_canonical_obs


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _state_key(e):
    return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())


def _canonical_training_sample(env, action_probs):
    obs = env.get_observation()
    mask = env.get_legal_action_mask()

    if env.current_player == 2:
        obs = get_canonical_obs(obs)
        mask = invert_action_array(mask)
        action_probs = invert_action_array(action_probs)

    return obs, mask, action_probs, env.current_player


def _heuristic_action(env):
    legal_actions = np.flatnonzero(env.get_legal_action_mask())
    if len(legal_actions) == 0:
        return 0

    player = env.current_player
    best_action = int(legal_actions[0])
    best_score = -float("inf")

    for act in legal_actions:
        sim_env = env.clone()
        _, reward, terminal, _ = sim_env.step(int(act))

        if terminal:
            score = 1000.0 if reward > 0 else -1000.0
        else:
            my_pos = sim_env.p1_pos if player == 1 else sim_env.p2_pos
            opp_pos = sim_env.p2_pos if player == 1 else sim_env.p1_pos
            my_target = 0 if player == 1 else 8
            opp_target = 8 if player == 1 else 0
            my_dist = sim_env._get_bfs_distance(my_pos, my_target)
            opp_dist = sim_env._get_bfs_distance(opp_pos, opp_target)
            score = (opp_dist - my_dist) + 0.03 * sim_env.walls_left[player]

        if score > best_score:
            best_score = score
            best_action = int(act)

    return best_action


def _worker_play_game(args):
    state_dict, config, game_idx = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WallzNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    env = WallzEnv()
    mcts = MCTS(model, num_simulations=config['mcts_simulations'])
    game_history = []
    seen_states = {_state_key(env): 1}

    terminal = False
    reward = 0.0
    step = 0

    while not terminal and step < config['max_steps']:
        # Do not downscale MCTS during self-play. Weak searches create weak targets.
        mcts.num_simulations = config['mcts_simulations']

        temp = 1.0 if step < config['temp_moves'] else 0.0
        action_probs = mcts.get_action_prob(env, temperature=temp)

        obs, mask, probs_for_training, player = _canonical_training_sample(env, action_probs)
        game_history.append((obs, mask, probs_for_training, player))
        game_history.append((
            flip_obs_horizontal(obs),
            flip_action_array_horizontal(mask),
            flip_action_array_horizontal(probs_for_training),
            player,
        ))

        legal_mask = env.get_legal_action_mask()
        legal_actions = np.flatnonzero(legal_mask)
        probs = np.zeros(TOTAL_ACTIONS, dtype=np.float64)
        probs[legal_actions] = action_probs[legal_actions]

        for act in legal_actions:
            sim_env = env.clone()
            sim_env.step(int(act))
            if seen_states.get(_state_key(sim_env), 0) >= config['rep_limit'] - 1:
                probs[act] = 0.0

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

        key = _state_key(env)
        seen_states[key] = seen_states.get(key, 0) + 1
        if not terminal and seen_states[key] >= config['rep_limit']:
            break

    # By default, do not train value on BFS/adjudicated pseudo-wins. It teaches "shortest path" instead of winning.
    if not terminal:
        if not config['train_on_adjudication']:
            return [], terminal, step

        p1_dist = env._get_bfs_distance(env.p1_pos, 0)
        p2_dist = env._get_bfs_distance(env.p2_pos, 8)
        if p1_dist == p2_dist:
            return [], terminal, step
        winner = 1 if p1_dist < p2_dist else 2
        base_reward = min(config['adjudication_reward_cap'], abs(p1_dist - p2_dist) * config['adjudication_reward_scale'])
    else:
        winner = 1 if env.current_player == 2 else 2
        base_reward = 1.0

    processed = []
    total_steps_in_game = len(game_history)
    for current_step_idx, (obs, mask, p, player) in enumerate(game_history):
        sign = 1.0 if player == winner else -1.0
        steps_to_end = total_steps_in_game - current_step_idx
        z = sign * base_reward * (config['value_discount'] ** steps_to_end)
        processed.append((obs, mask, p, z))

    return processed, terminal, step


def _worker_eval_game(args):
    state_dict, config, game_idx = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WallzNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    env = WallzEnv()
    mcts = MCTS(model, num_simulations=config['eval_mcts_simulations'])
    seen_states = {_state_key(env): 1}
    terminal = False
    reward = 0.0
    step = 0

    model_player = 1 if game_idx % 2 == 0 else 2

    while not terminal and step < config['eval_max_steps']:
        if env.current_player == model_player:
            action_probs = mcts.get_action_prob(env, temperature=0.0)
            action = int(np.argmax(action_probs))
        else:
            action = _heuristic_action(env)

        _, reward, terminal, _ = env.step(action)
        step += 1

        key = _state_key(env)
        seen_states[key] = seen_states.get(key, 0) + 1
        if not terminal and seen_states[key] >= config['rep_limit']:
            break

    if terminal:
        winner = 1 if env.current_player == 2 else 2
    else:
        p1_dist = env._get_bfs_distance(env.p1_pos, 0)
        p2_dist = env._get_bfs_distance(env.p2_pos, 8)
        winner = 0 if p1_dist == p2_dist else (1 if p1_dist < p2_dist else 2)

    return winner == model_player, terminal, step


class AlphaZeroTrainer:
    def __init__(self):
        self.cpu_threads = env_int("AZ_TORCH_THREADS", 20)
        torch.set_num_threads(self.cpu_threads)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = WallzNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.epochs = env_int("AZ_EPOCHS", 300)
        self.games_per_epoch = env_int("AZ_GAMES_PER_EPOCH", 100)

        self.mcts_simulations = 50
        self.temperature_moves = 20

        self.batch_size = env_int("AZ_BATCH_SIZE", 256)
        self.save_every = env_int("AZ_SAVE_EVERY", 1)
        self.min_terminal_games = env_int("AZ_MIN_TERMINAL_GAMES", max(1, self.games_per_epoch // 4))
        self.max_self_play_games = env_int("AZ_MAX_SELF_PLAY_GAMES", self.games_per_epoch * 3)
        self.max_steps_per_game = env_int("AZ_MAX_STEPS_PER_GAME", 200)
        self.repetition_limit = env_int("AZ_REPETITION_LIMIT", 3)
        self.show_progress = env_flag("AZ_PROGRESS", True)
        self.train_on_adjudication = env_flag("AZ_TRAIN_ON_ADJUDICATION", False)
        self.adjudication_reward_scale = env_float("AZ_ADJUDICATION_REWARD_SCALE", 0.05)
        self.adjudication_reward_cap = env_float("AZ_ADJUDICATION_REWARD_CAP", 0.25)
        self.value_discount = env_float("AZ_VALUE_DISCOUNT", 0.99)

        self.eval_every = env_int("AZ_EVAL_EVERY", 10)
        self.eval_games = env_int("AZ_EVAL_GAMES", 20)
        self.eval_mcts_simulations = env_int("AZ_EVAL_MCTS_SIMULATIONS", 100)
        self.best_win_rate = 0.0

        self.replay_buffer = deque(maxlen=200000)
        self.checkpoint_dir = PACKAGE_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = self._load_latest_checkpoint() + 1
        self.current_epoch = self.start_epoch - 1

    def _load_latest_checkpoint(self) -> int:
        checkpoints = []
        for pattern in ("alphazero_epoch_*.pt", "alphazero_interrupt_epoch_*.pt"):
            for path in self.checkpoint_dir.glob(pattern):
                match = re.search(r"epoch_(\d+)", path.name)
                if match:
                    checkpoints.append((int(match.group(1)), path.stat().st_mtime, path))

        if not checkpoints:
            print("No AlphaZero checkpoint found. Starting from scratch.")
            return 0

        epoch, _, path = max(checkpoints, key=lambda item: (item[0], item[1]))
        print(f"♻️ Loading AlphaZero checkpoint: {path}")
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Resuming after epoch {epoch}.")
            return epoch
        except RuntimeError:
            print("⚠️ Checkpoint architecture mismatch! Starting from scratch.")
            return 0

    def _apply_dynamic_scheduler(self, epoch):
        if epoch <= 50:
            current_lr = 1e-3
            self.mcts_simulations = 50
            self.temperature_moves = 20
        elif epoch <= 150:
            current_lr = 5e-4
            self.mcts_simulations = 100
            self.temperature_moves = 12
        else:
            current_lr = 1e-4
            self.mcts_simulations = 150
            self.temperature_moves = 6

        for param_group in self.optimizer.param_groups:
            if param_group['lr'] != current_lr:
                param_group['lr'] = current_lr
                tqdm.write(f"⚙️ Dynamic Scheduler Triggered! Epoch {epoch}: LR -> {current_lr}, MCTS -> {self.mcts_simulations}, TempMoves -> {self.temperature_moves}")

    def save_checkpoint(self, epoch: int, interrupted: bool = False):
        name = f"alphazero_interrupt_epoch_{epoch}.pt" if interrupted else f"alphazero_epoch_{epoch}.pt"
        path = self.checkpoint_dir / name
        torch.save(self.model.state_dict(), path)
        torch.save(self.model.state_dict(), self.checkpoint_dir / "alphazero_latest.pt")
        tqdm.write(f"💾 Saved ResNet checkpoint to {path}")

    def save_best_checkpoint(self, epoch: int, win_rate: float):
        path = self.checkpoint_dir / "alphazero_best.pt"
        torch.save(self.model.state_dict(), path)
        tqdm.write(f"🏆 New best checkpoint at epoch {epoch}: win_rate={win_rate:.3f}")

    def _self_play_config(self):
        return {
            'mcts_simulations': self.mcts_simulations,
            'max_steps': self.max_steps_per_game,
            'temp_moves': self.temperature_moves,
            'rep_limit': self.repetition_limit,
            'train_on_adjudication': self.train_on_adjudication,
            'adjudication_reward_scale': self.adjudication_reward_scale,
            'adjudication_reward_cap': self.adjudication_reward_cap,
            'value_discount': self.value_discount,
        }

    def self_play(self):
        self.model.eval()
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        config = self._self_play_config()

        num_workers = min(os.cpu_count() or 4, 10)
        completed, adjudicated, total_steps, attempted = 0, 0, 0, 0
        next_game_idx = 0

        while attempted < self.max_self_play_games:
            remaining_budget = self.max_self_play_games - attempted
            batch_games = min(self.games_per_epoch, remaining_budget)
            args_list = [(state_dict, config, next_game_idx + i) for i in range(batch_games)]
            next_game_idx += batch_games
            attempted += batch_games

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_worker_play_game, args) for args in args_list]
                iterator = concurrent.futures.as_completed(futures)
                if self.show_progress:
                    iterator = tqdm(iterator, total=len(futures), desc="Parallel Self-Play")

                for future in iterator:
                    processed_history, terminal, steps = future.result()
                    self.replay_buffer.extend(processed_history)
                    total_steps += steps
                    if terminal:
                        completed += 1
                    else:
                        adjudicated += 1

            if completed >= self.min_terminal_games and attempted >= self.games_per_epoch:
                break

            tqdm.write(f"⚠️ Only {completed} terminal games after {attempted} attempts; collecting more self-play games...")

        counted = completed + adjudicated
        return {
            "completed_games": completed,
            "adjudicated_games": adjudicated,
            "attempted_games": attempted,
            "avg_steps": total_steps / counted if counted else 0,
            "replay_buffer": len(self.replay_buffer),
        }

    def train_network(self, terminal_games_this_epoch: int):
        if terminal_games_this_epoch < self.min_terminal_games:
            tqdm.write(f"⚠️ Skipping train step: terminal games {terminal_games_this_epoch} < minimum {self.min_terminal_games}")
            return None

        if len(self.replay_buffer) < self.batch_size:
            return None

        self.model.train()
        training_steps = min(1000, max(50, len(self.replay_buffer) // self.batch_size))
        total_policy_loss, total_value_loss = 0, 0

        for _ in range(training_steps):
            batch = random.sample(self.replay_buffer, self.batch_size)
            state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
            mask_batch = torch.BoolTensor(np.array([x[1] for x in batch])).to(self.device)
            prob_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
            value_batch = torch.FloatTensor(np.array([x[3] for x in batch]).astype(np.float32)).unsqueeze(1).to(self.device)

            logits, values = self.model(state_batch, action_mask=mask_batch)

            policy_loss = -torch.sum(prob_batch * F.log_softmax(logits, dim=1), dim=1).mean()
            value_loss = F.mse_loss(values, value_batch)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            "policy_loss": total_policy_loss / training_steps,
            "value_loss": total_value_loss / training_steps,
            "total_loss": (total_policy_loss + total_value_loss) / training_steps,
            "training_steps": training_steps,
        }

    def evaluate(self):
        if self.eval_every <= 0 or self.eval_games <= 0:
            return None

        self.model.eval()
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        config = {
            'eval_mcts_simulations': self.eval_mcts_simulations,
            'eval_max_steps': self.max_steps_per_game,
            'rep_limit': self.repetition_limit,
        }

        num_workers = min(os.cpu_count() or 4, 10)
        args_list = [(state_dict, config, i) for i in range(self.eval_games)]
        wins, terminals, total_steps = 0, 0, 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_eval_game, args) for args in args_list]
            iterator = concurrent.futures.as_completed(futures)
            if self.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Evaluation")

            for future in iterator:
                won, terminal, steps = future.result()
                wins += int(won)
                terminals += int(terminal)
                total_steps += steps

        return {
            "win_rate": wins / self.eval_games,
            "terminal_rate": terminals / self.eval_games,
            "avg_steps": total_steps / self.eval_games,
        }

    def learn(self):
        final_epoch_exclusive = self.start_epoch + self.epochs
        epoch_iter = tqdm(
            range(self.start_epoch, final_epoch_exclusive),
            total=(final_epoch_exclusive - self.start_epoch),
            desc="AlphaZero ResNet",
            dynamic_ncols=True,
        )

        for epoch in epoch_iter:
            self.current_epoch = epoch
            self._apply_dynamic_scheduler(epoch)

            stats = self.self_play()
            losses = self.train_network(stats["completed_games"])

            postfix = {
                "games": stats["completed_games"],
                "attempts": stats["attempted_games"],
                "replay": stats["replay_buffer"],
            }
            if losses is not None:
                postfix["loss"] = f"{losses['total_loss']:.3f}"
                postfix["steps"] = losses["training_steps"]
            epoch_iter.set_postfix(postfix, refresh=True)

            if self.eval_every > 0 and epoch % self.eval_every == 0:
                eval_stats = self.evaluate()
                if eval_stats is not None:
                    tqdm.write(
                        f"📊 Eval epoch {epoch}: win_rate={eval_stats['win_rate']:.3f}, "
                        f"terminal_rate={eval_stats['terminal_rate']:.3f}, avg_steps={eval_stats['avg_steps']:.1f}"
                    )
                    if eval_stats["win_rate"] > self.best_win_rate:
                        self.best_win_rate = eval_stats["win_rate"]
                        self.save_best_checkpoint(epoch, self.best_win_rate)

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    trainer = AlphaZeroTrainer()
    try:
        trainer.learn()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving ResNet checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, interrupted=True)
