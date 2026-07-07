import sys
import asyncio
import numpy as np
import torch
import re
from pathlib import Path
from playwright.async_api import async_playwright

# Добавляем корень проекта в sys.path для импорта модулей
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS
from wallz_v2.env.wallz_env import WallzEnv
from wallz_v2.env.action_space import action_to_move, move_to_action

class WallzAssistant:
    def __init__(self, model_path):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading AlphaZero model on: {self.device}")
        
        self.model = WallzNet(num_channels=8) 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.mcts = MCTS(self.model, num_simulations=200) 

    async def extract_board_state(self, page):
        env = WallzEnv()
        seen_states = {}
        def state_key(e): return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())
        seen_states[state_key(env)] = 1
        
        try:
            elements = await page.locator('ol > li').all_inner_texts()
            history_text = " ".join(elements).lower()
            
            if not history_text:
                return env, seen_states

            # Regex to find all valid Quoridor moves
            moves = re.findall(r'\b([a-i])([1-9])([hv]?)\b', history_text)
            
            # Universal Mathematical Map
            col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
            row_map = {'9': 0, '8': 1, '7': 2, '6': 3, '5': 4, '4': 5, '3': 6, '2': 7, '1': 8}

            for m in moves:
                c_char, r_char, w_char = m
                col_idx = col_map[c_char]
                row_idx = row_map[r_char]
                
                if not w_char:
                    action = move_to_action('MOVE', row_idx, col_idx)
                else:
                    if w_char == 'h':
                        wall_r = row_idx
                        wall_c = col_idx
                    elif w_char == 'v':
                        wall_r = row_idx - 1
                        wall_c = col_idx - 1
                        
                    if wall_r < 0 or wall_r > 7 or wall_c < 0 or wall_c > 7:
                        continue
                        
                    action = move_to_action(f'WALL_{w_char.upper()}', wall_r, wall_c)
                        
                env.step(action)
                key = state_key(env)
                seen_states[key] = seen_states.get(key, 0) + 1
                
        except Exception as e:
            print(f"❌ Error parsing board history: {e}")
            
        return env, seen_states

    async def run(self):
        print("🚀 Booting Auto-Stealth Visual Assistant with Persistent Profile...")
        
        profile_dir = ROOT_DIR / "wallz_v2" / "browser_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=False, 
                args=['--disable-blink-features=AutomationControlled'],
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto("https://wallz.gg")
            print("\n✅ Browser open! Auto-highlighting is enabled.")
            
            status_js = """
            () => {
                if (!document.getElementById('az-status')) {
                    const status = document.createElement('div');
                    status.id = 'az-status';
                    status.style.position = 'fixed';
                    status.style.bottom = '10px';
                    status.style.right = '10px';
                    status.style.zIndex = '999999';
                    status.style.backgroundColor = 'rgba(15, 23, 42, 0.7)';
                    status.style.color = '#94a3b8';
                    status.style.padding = '8px 12px';
                    status.style.borderRadius = '8px';
                    status.style.fontFamily = 'monospace';
                    status.style.fontSize = '12px';
                    status.style.pointerEvents = 'none';
                    status.innerText = '🟢 AI Active';
                    document.body.appendChild(status);
                }
            }
            """
            
            last_processed_moves = -1
            
            col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
            row_map = {'9': 0, '8': 1, '7': 2, '6': 3, '5': 4, '4': 5, '3': 6, '2': 7, '1': 8}
            col_map_inv = {v: k for k, v in col_map.items()}
            row_map_inv = {v: k for k, v in row_map.items()}

            while True:
                try:
                    await page.evaluate(status_js)
                    is_my_turn = await page.locator("text='Your turn'").is_visible()
                    
                    if is_my_turn:
                        history_text = " ".join(await page.locator('ol > li').all_inner_texts()).lower()
                        current_moves = len(re.findall(r'\b[a-i][1-9][hv]?\b', history_text))
                        
                        if current_moves != last_processed_moves:
                            await page.evaluate("document.getElementById('az-status').innerText = '⏳ Thinking...'")
                            
                            env, seen_states = await self.extract_board_state(page)
                            action_probs = self.mcts.get_action_prob(env, temperature=0.2)
                            
                            legal_mask = env.get_legal_action_mask()
                            legal_actions = np.flatnonzero(legal_mask)
                            probs = np.zeros(209)
                            probs[legal_actions] = action_probs[legal_actions]
                            
                            def state_key(e): return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())
                            
                            for act in legal_actions:
                                saved_p1, saved_p2, saved_cp = env.p1_pos, env.p2_pos, env.current_player
                                saved_wl = env.walls_left.copy()
                                saved_hw, saved_vw = env.h_walls.copy(), env.v_walls.copy()
                                
                                env.step(int(act))
                                if seen_states.get(state_key(env), 0) >= 1:
                                    probs[act] = 0.0
                                    
                                env.p1_pos, env.p2_pos, env.current_player = saved_p1, saved_p2, saved_cp
                                env.walls_left = saved_wl
                                env.h_walls, env.v_walls = saved_hw, saved_vw

                            if probs.sum() > 0:
                                best_action = int(np.argmax(probs))
                            else:
                                best_action = int(np.argmax(action_probs))
                            
                            move_type, (r, c) = action_to_move(best_action)
                            
                            # Convert internal coords back to notation chars for the visualizer
                            if move_type == 'MOVE':
                                hint_c_char = col_map_inv[c]
                                hint_r_char = row_map_inv[r]
                            elif move_type == 'WALL_V':
                                hint_c_char = col_map_inv[c + 1]
                                hint_r_char = row_map_inv[r + 1]
                            elif move_type == 'WALL_H':
                                hint_c_char = col_map_inv[c]
                                hint_r_char = row_map_inv[r]
                            
                            highlight_js = f"""
                            () => {{
                                const svg = document.querySelector('svg[aria-label="Wallz board"]');
                                if (!svg) return;

                                let oldHighlight = document.getElementById('az-visual-hint');
                                if (oldHighlight) oldHighlight.remove();

                                const highlight = document.createElementNS("http://www.w3.org/2000/svg", "g");
                                highlight.id = 'az-visual-hint';
                                highlight.style.pointerEvents = 'none';

                                const moveType = '{move_type}';
                                const cChar = '{hint_c_char}';
                                const rChar = '{hint_r_char}';

                                // Dynamically read the DOM text to find out if the board is flipped
                                const cols = Array.from(document.querySelectorAll('text[y="660"]')).map(t => t.textContent.trim().toLowerCase());
                                const rows = Array.from(document.querySelectorAll('text[x="-24"]')).map(t => t.textContent.trim().toLowerCase());

                                const visual_c_idx = cols.indexOf(cChar);
                                const visual_r_idx = rows.indexOf(rChar);

                                if (visual_c_idx === -1 || visual_r_idx === -1) return;

                                let vis_c, vis_r;
                                let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                                
                                if (moveType === 'MOVE') {{
                                    vis_c = visual_c_idx;
                                    vis_r = visual_r_idx;
                                    rect.setAttribute('x', vis_c * 72);
                                    rect.setAttribute('y', vis_r * 72);
                                    rect.setAttribute('width', 60);
                                    rect.setAttribute('height', 60);
                                    rect.setAttribute('rx', 9);
                                    rect.setAttribute('fill', 'rgba(56, 189, 248, 0.4)');
                                    rect.setAttribute('stroke', '#38bdf8');
                                    rect.setAttribute('stroke-width', 4);
                                }} else if (moveType === 'WALL_V') {{
                                    vis_c = visual_c_idx - 1;
                                    vis_r = visual_r_idx - 1;
                                    rect.setAttribute('x', (vis_c * 72) + 60);
                                    rect.setAttribute('y', vis_r * 72);
                                    rect.setAttribute('width', 12);
                                    rect.setAttribute('height', 132);
                                    rect.setAttribute('rx', 5);
                                    rect.setAttribute('fill', 'rgba(250, 204, 21, 0.8)');
                                    rect.setAttribute('stroke', '#facc15');
                                }} else if (moveType === 'WALL_H') {{
                                    vis_c = visual_c_idx;
                                    vis_r = visual_r_idx;
                                    rect.setAttribute('x', vis_c * 72);
                                    rect.setAttribute('y', (vis_r * 72) + 60);
                                    rect.setAttribute('width', 132);
                                    rect.setAttribute('height', 12);
                                    rect.setAttribute('rx', 5);
                                    rect.setAttribute('fill', 'rgba(250, 204, 21, 0.8)');
                                    rect.setAttribute('stroke', '#facc15');
                                }}

                                const animate = document.createElementNS("http://www.w3.org/2000/svg", "animate");
                                animate.setAttribute('attributeName', 'opacity');
                                animate.setAttribute('values', '0.3; 1; 0.3');
                                animate.setAttribute('dur', '1s');
                                animate.setAttribute('repeatCount', 'indefinite');
                                rect.appendChild(animate);

                                highlight.appendChild(rect);
                                svg.appendChild(highlight);
                            }}
                            """
                            await page.evaluate(highlight_js)
                            await page.evaluate("document.getElementById('az-status').innerText = '👑 Move Highlighted!'")
                            
                            # Print standard notation to the terminal so you know exactly what is happening
                            print(f"Move {current_moves + 1} -> Highlighted {move_type} at {hint_c_char}{hint_r_char}")
                            
                            last_processed_moves = current_moves
                    else:
                        await page.evaluate("""() => {
                            let oldHighlight = document.getElementById('az-visual-hint');
                            if (oldHighlight) oldHighlight.remove();
                            let status = document.getElementById('az-status');
                            if (status && status.innerText !== '🟢 AI Active') status.innerText = '🟢 AI Active';
                        }""")
                            
                except Exception as e:
                    pass
                
                await asyncio.sleep(0.5)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    CHECKPOINT = ROOT_DIR / "wallz_v2" / "checkpoints" / "alphazero_latest.pt"
    
    if not CHECKPOINT.exists():
        print(f"Model not found at {CHECKPOINT}")
        sys.exit(1)
        
    assistant = WallzAssistant(CHECKPOINT)
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")