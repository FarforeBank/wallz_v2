import re
from pathlib import Path

file_path = Path('scripts/train_alphazero.py')
code = file_path.read_text()

# 1. Define the standalone worker function that runs parallel games
worker_code = """
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
"""

# Insert the worker code right before the Trainer class
code = code.replace("class AlphaZeroTrainer:", worker_code + "\nclass AlphaZeroTrainer:")

# 2. Rewrite the self_play method to use the Process Pool
new_self_play = """    def self_play(self):
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

        print(f"\\n  Spawning {num_workers} parallel workers for {self.games_per_epoch} self-play games...")

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

    def train_network"""

# Inject the parallel method
code = re.sub(r'    def self_play\(self\):.*?    def train_network', new_self_play, code, flags=re.DOTALL)

# 3. Add the required Apple Silicon 'spawn' safeguard to the main block
spawn_safeguard = """if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass"""
code = code.replace("if __name__ == '__main__':", spawn_safeguard)

file_path.write_text(code)
print("✅ Multiprocessing enabled! Self-play is now distributing workloads across CPU cores.")
