import numpy as np
from collections import deque
from .action_space import TOTAL_ACTIONS, action_to_move

ORTHOGONAL_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class WallzEnv:
    def __init__(self):
        self.board_size = 9
        self.reset()

    def reset(self):
        self.h_walls = np.zeros((8, 8), dtype=bool)
        self.v_walls = np.zeros((8, 8), dtype=bool)

        self.p1_pos = (4, 8)  # Player 1 starts at bottom row (x=4, y=8)
        self.p2_pos = (4, 0)  # Player 2 starts at top row (x=4, y=0)

        self.current_player = 1
        self.walls_left = {1: 10, 2: 10}
        return self.get_observation(), self.get_legal_action_mask()

    def _in_bounds(self, x, y):
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def has_wall_between(self, x1, y1, x2, y2):
        if not self._in_bounds(x1, y1) or not self._in_bounds(x2, y2):
            return True
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return True

        if x1 == x2:  # Vertical pawn movement: check horizontal blocking barriers
            y_min = min(y1, y2)
            if x1 < 8 and self.h_walls[y_min, x1]:
                return True
            if x1 > 0 and self.h_walls[y_min, x1 - 1]:
                return True
        elif y1 == y2:  # Horizontal pawn movement: check vertical blocking barriers
            x_min = min(x1, x2)
            if y1 < 8 and self.v_walls[y1, x_min]:
                return True
            if y1 > 0 and self.v_walls[y1 - 1, x_min]:
                return True
        return False

    def _get_valid_moves(self, player_id):
        """Your full engine jump and diagonal tracking logic integrated into (row, col) coordinates."""
        cx, cy = self.p1_pos if player_id == 1 else self.p2_pos
        ox, oy = self.p2_pos if player_id == 1 else self.p1_pos
        moves = []

        for dx, dy in ORTHOGONAL_DIRS:
            ax, ay = cx + dx, cy + dy
            if not self._in_bounds(ax, ay) or self.has_wall_between(cx, cy, ax, ay):
                continue

            if (ax, ay) != (ox, oy):
                moves.append((ax, ay))
                continue

            # Straight jump sequence if opponent is blocking the path
            jx, jy = ax + dx, ay + dy
            if self._in_bounds(jx, jy) and not self.has_wall_between(ax, ay, jx, jy):
                moves.append((jx, jy))
                continue

            # Diagonal bypass rule when straight jumps are obstructed
            perpendicular = [(-1, 0), (1, 0)] if dx == 0 else [(0, -1), (0, 1)]
            for pdx, pdy in perpendicular:
                tx, ty = ax + pdx, ay + pdy
                if self._in_bounds(tx, ty) and not self.has_wall_between(ax, ay, tx, ty):
                    moves.append((tx, ty))

        # Re-map (x, y) engine tracking into standard internal structural matrix layout: (row=y, col=x)
        seen = set()
        unique_rc_moves = []
        for x, y in moves:
            rc = (y, x)
            if rc not in seen:
                unique_rc_moves.append(rc)
                seen.add(rc)
        return unique_rc_moves

    def _get_bfs_distance(self, start_pos, target_row):
        queue = deque([(start_pos[0], start_pos[1], 0)])
        visited = {start_pos}

        while queue:
            x, y, dist = queue.popleft()
            if y == target_row:
                return dist

            for dx, dy in ORTHOGONAL_DIRS:
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny) and not self.has_wall_between(x, y, nx, ny):
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        return 999

    def _has_path(self, player_id):
        start = self.p1_pos if player_id == 1 else self.p2_pos
        target = 0 if player_id == 1 else 8
        return self._get_bfs_distance(start, target) != 999

    def _wall_slot_is_free(self, r, c, orientation):
        if r < 0 or r > 7 or c < 0 or c > 7:
            return False
        if orientation == "H":
            if self.h_walls[r, c]: return False
            if c > 0 and self.h_walls[r, c - 1]: return False
            if c < 7 and self.h_walls[r, c + 1]: return False
            if self.v_walls[r, c]: return False
            return True
        else:
            if self.v_walls[r, c]: return False
            if r > 0 and self.v_walls[r - 1, c]: return False
            if r < 7 and self.v_walls[r + 1, c]: return False
            if self.h_walls[r, c]: return False
            return True

    def _can_place_wall(self, r, c, orientation):
        if not self._wall_slot_is_free(r, c, orientation):
            return False

        # Temporarily anchor the wall matrix configuration to validate valid pathways
        if orientation == "H":
            self.h_walls[r, c] = True
            valid = self._has_path(1) and self._has_path(2)
            self.h_walls[r, c] = False
        else:
            self.v_walls[r, c] = True
            valid = self._has_path(1) and self._has_path(2)
            self.v_walls[r, c] = False
        return valid

    def get_observation(self):
        """Constructs an 8-channel full spatial matrix tensor plane (8, 9, 9) representing absolute space."""
        obs = np.zeros((8, 9, 9), dtype=np.float32)
        
        # Setup views centered cleanly around current turn perspectivism
        cp = self.current_player
        curr_pos = (self.p1_pos[1], self.p1_pos[0]) if cp == 1 else (self.p2_pos[1], self.p2_pos[0])
        opp_pos = (self.p2_pos[1], self.p2_pos[0]) if cp == 1 else (self.p1_pos[1], self.p1_pos[0])
        
        obs[0, curr_pos[0], curr_pos[1]] = 1.0  # CH 0: Current Player coordinates
        obs[1, opp_pos[0], opp_pos[1]] = 1.0    # CH 1: Opponent coordinates
        
        obs[2, :8, :8] = self.h_walls           # CH 2: Horizontal layouts mapped
        obs[3, :8, :8] = self.v_walls           # CH 3: Vertical layouts mapped
        
        obs[4, (0 if cp == 1 else 8), :] = 1.0  # CH 4: Self goal lane
        obs[5, (8 if cp == 1 else 0), :] = 1.0  # CH 5: Target opponent goal lane
        
        obs[6, :, :] = self.walls_left[cp] / 10.0      # CH 6: Player wall balance plane
        obs[7, :, :] = self.walls_left[3 - cp] / 10.0  # CH 7: Enemy wall balance plane
        return obs

    def get_legal_action_mask(self):
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
        
        # 1. Map absolute valid pawn row-col destinations (0-80)
        valid_moves = self._get_valid_moves(self.current_player)
        for r, c in valid_moves:
            mask[r * 9 + c] = True
            
        # 2. Map absolute valid wall coordinate anchors if budget permits
        if self.walls_left[self.current_player] > 0:
            for r in range(8):
                for c in range(8):
                    if self._can_place_wall(r, c, "H"):
                        mask[81 + (r * 8 + c)] = True
                    if self._can_place_wall(r, c, "V"):
                        mask[145 + (r * 8 + c)] = True
        return mask

    def step(self, action: int):
        move_type, (r, c) = action_to_move(action)
        
        if move_type == 'MOVE':
            # Decode matrix indices (row=r, col=c) back into engine (x=col, y=row) layout
            new_pos = (c, r)
            if self.current_player == 1:
                self.p1_pos = new_pos
            else:
                self.p2_pos = new_pos
        elif move_type == 'WALL_H':
            self.h_walls[r, c] = True
            self.walls_left[self.current_player] -= 1
        elif move_type == 'WALL_V':
            self.v_walls[r, c] = True
            self.walls_left[self.current_player] -= 1

        reward = 0.0
        terminal = False
        
        # Check finish line benchmarks
        if self.p1_pos[1] == 0:
            reward = 1.0 if self.current_player == 1 else -1.0
            terminal = True
        elif self.p2_pos[1] == 8:
            reward = 1.0 if self.current_player == 2 else -1.0
            terminal = True

        self.current_player = 3 - self.current_player
        return self.get_observation(), reward, terminal, self.get_legal_action_mask()
