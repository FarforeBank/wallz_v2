import sys
from pathlib import Path
import numpy as np

# Add the project root to the path so we can import our modules
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.env.action_space import action_to_move

def print_board(env):
    """A simple CLI visualization of the board state."""
    print(f"\n--- Player {env.current_player}'s Turn ---")
    print(f"P1 (Bottom) Walls Left: {env.walls_left[1]} | Position: {env.p1_pos}")
    print(f"P2 (Top) Walls Left: {env.walls_left[2]} | Position: {env.p2_pos}")
    
    for r in range(9):
        row_str = ""
        for c in range(9):
            if (c, r) == env.p1_pos:
                row_str += " 1 "
            elif (c, r) == env.p2_pos:
                row_str += " 2 "
            else:
                row_str += " . "
            
            # Print vertical wall if it exists to the right of this cell
            if c < 8 and r < 8 and env.v_walls[r, c]:
                row_str += "|"
            elif c < 8 and r > 0 and env.v_walls[r-1, c]:
                row_str += "|"
            else:
                row_str += " "
        print(row_str)
        
        # Print horizontal walls below the current row
        if r < 8:
            h_row_str = ""
            for c in range(8):
                if env.h_walls[r, c]:
                    h_row_str += "--- "
                else:
                    h_row_str += "    "
            print(h_row_str)
    print("--------------------------\n")

def main():
    env = WallzEnv()
    obs, mask = env.reset()
    
    print("Initial State:")
    print_board(env)
    
    step_count = 0
    terminal = False
    
    while not terminal and step_count < 100:
        step_count += 1
        
        # Get indices of all legal actions from the mask
        legal_actions = np.where(mask == True)[0]
        
        if len(legal_actions) == 0:
            print("ERROR: No legal actions available! Game is stuck.")
            break
            
        # Choose a random legal action
        chosen_action = np.random.choice(legal_actions)
        move_type, (r, c) = action_to_move(chosen_action)
        
        print(f"Step {step_count}: Player {env.current_player} chooses {move_type} at ({r}, {c})")
        
        # Take the step
        obs, reward, terminal, mask = env.step(chosen_action)
        print_board(env)
        
        if terminal:
            winner = 1 if reward == 1.0 and env.current_player == 2 else 2 # Current player has already swapped
            print(f"Game Over! Player {winner} wins!")
            break

if __name__ == '__main__':
    main()
