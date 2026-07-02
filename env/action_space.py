import numpy as np

# Spatial Absolute Coordinates Action Space Constants
NUM_SQUARES = 81       # 9x9 grid for direct pawn target coordinates (0-80)
NUM_H_WALLS = 64       # 8x8 grid for horizontal wall intersection anchors (81-144)
NUM_V_WALLS = 64       # 8x8 grid for vertical wall intersection anchors (145-208)
TOTAL_ACTIONS = NUM_SQUARES + NUM_H_WALLS + NUM_V_WALLS

def action_to_move(action: int):
    """Decodes a flat absolute action integer to a structural game command."""
    if action < NUM_SQUARES:
        return 'MOVE', (action // 9, action % 9)
    elif action < NUM_SQUARES + NUM_H_WALLS:
        idx = action - NUM_SQUARES
        return 'WALL_H', (idx // 8, idx % 8)
    else:
        idx = action - (NUM_SQUARES + NUM_H_WALLS)
        return 'WALL_V', (idx // 8, idx % 8)

def move_to_action(move_type: str, r: int, c: int) -> int:
    """Encodes a specific coordinate game choice back into a flat network action."""
    if move_type == 'MOVE':
        return r * 9 + c
    elif move_type == 'WALL_H':
        return NUM_SQUARES + (r * 8 + c)
    elif move_type == 'WALL_V':
        return NUM_SQUARES + NUM_H_WALLS + (r * 8 + c)
    raise ValueError(f"Unknown move type: {move_type}")
