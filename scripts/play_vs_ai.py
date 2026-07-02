import sys
from pathlib import Path
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sb3_contrib import MaskablePPO
from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.env.action_space import action_to_move, move_to_action
from wallz_v2.scripts.test_env import print_board

def get_human_action(env, mask):
    """Prompts the user for a valid action."""
    while True:
        try:
            print("\nCommands:")
            print("  m r c   -> Move to row r, col c (e.g., 'm 3 4')")
            print("  h r c   -> Place Horizontal wall at row r, col c (e.g., 'h 5 2')")
            print("  v r c   -> Place Vertical wall at row r, col c (e.g., 'v 1 1')")
            
            cmd = input(f"Player {env.current_player} Your move: ").strip().lower().split()
            if len(cmd) != 3:
                print("Invalid format. Use: type row col")
                continue
                
            action_type, r, c = cmd[0], int(cmd[1]), int(cmd[2])
            
            if action_type == 'm':
                action = move_to_action('MOVE', r, c)
            elif action_type == 'h':
                action = move_to_action('WALL_H', r, c)
            elif action_type == 'v':
                action = move_to_action('WALL_V', r, c)
            else:
                print("Invalid command. Use 'm', 'h', or 'v'.")
                continue
                
            if action < 0 or action >= len(mask) or not mask[action]:
                print("❌ ILLEGAL MOVE! You cannot do that right now (blocked or out of walls).")
                continue
                
            return action
        except Exception as e:
            print("Invalid input. Please enter numbers for row and col.")

def main():
    model_path = ROOT_DIR / "wallz_v2" / "checkpoints" / "ppo_v2_model.zip"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}!")
        print("Wait for the training script to finish and save the model.")
        return

    print("Loading Trained AI...")
    model = MaskablePPO.load(model_path, device="cpu")
    
    env = WallzEnv()
    obs, mask = env.reset()
    
    # Ask who goes first
    human_player = input("Do you want to be Player 1 (bottom, moves first) or Player 2 (top)? [1/2]: ").strip()
    human_player = 1 if human_player == '1' else 2
    
    terminal = False
    
    while not terminal:
        print_board(env)
        
        if env.current_player == human_player:
            action = get_human_action(env, mask)
        else:
            print("\n🤖 AI is thinking...")
            # Deterministic=True makes the AI pick its absolute best move rather than exploring
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            move_type, (r, c) = action_to_move(action)
            print(f"🤖 AI played: {move_type} at ({r}, {c})")
            
        obs, reward, terminal, mask = env.step(action)
        
    print_board(env)
    winner = 1 if reward == 1.0 and env.current_player == 2 else 2
    
    if winner == human_player:
        print("\n🎉 YOU BEAT THE AI! 🎉")
    else:
        print("\n💀 THE AI WON! 💀")

if __name__ == '__main__':
    main()
