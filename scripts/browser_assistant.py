import sys
import re
import asyncio
import numpy as np
import torch
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
        # Автоопределение Apple Silicon (MPS), CUDA или CPU
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading AlphaZero model on: {self.device}")
        
        # Загружаем чекпоинт
        self.model = WallzNet(num_channels=8) 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Инициализация MCTS
        self.mcts = MCTS(self.model, num_simulations=50) 

    async def extract_board_state(self, page):
        env = WallzEnv()
        seen_states = {}
        def state_key(e): return (e.p1_pos, e.p2_pos, e.current_player, e.walls_left[1], e.walls_left[2], e.h_walls.tobytes(), e.v_walls.tobytes())
        seen_states[state_key(env)] = 1
        
        try:
            # Safely grab all raw text from the history list, ignoring CSS/HTML tags
            elements = await page.locator('ol > li').all_inner_texts()
            history_text = " ".join(elements).lower()
            
            if not history_text:
                return env, seen_states

            # Regex to find all valid Quoridor moves: e.g., 'e2', 'e8v', 'c5h'
            moves = re.findall(r'([a-i])([1-9])([hv]?)', history_text)
            
            col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
            row_map = {'9': 0, '8': 1, '7': 2, '6': 3, '5': 4, '4': 5, '3': 6, '2': 7, '1': 8}

            for m in moves:
                c_char, r_char, w_char = m
                col_idx = col_map[c_char]
                row_idx = row_map[r_char]
                
                if not w_char:
                    action = move_to_action('MOVE', row_idx, col_idx)
                else:
                    # Direct coordinate mapping (Reverted the math bug)
                    wall_r = row_idx
                    wall_c = col_idx
                    if wall_r < 0 or wall_r > 7 or wall_c < 0 or wall_c > 7:
                        continue
                    if w_char == 'h':
                        action = move_to_action('WALL_H', wall_r, wall_c)
                    elif w_char == 'v':
                        action = move_to_action('WALL_V', wall_r, wall_c)
                        
                env.step(action)
                key = state_key(env)
                seen_states[key] = seen_states.get(key, 0) + 1
                
        except Exception as e:
            print(f"❌ Error parsing board history: {e}")
            
        return env, seen_states

            # 2. Маппинг координат Wallz.gg (i->a, 9->1)
            col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
            row_map = {'9': 0, '8': 1, '7': 2, '6': 3, '5': 4, '4': 5, '3': 6, '2': 7, '1': 8}

            # 3. Проигрываем историю для синхронизации
            for move_str in move_elements:
                move_str = move_str.strip().lower()
                if not move_str: 
                    continue

                col_idx = col_map[move_str[0]]
                row_idx = row_map[move_str[1]]
                
                if len(move_str) == 2:
                    # Для пешек координаты прямые (0-8)
                    action = move_to_action('MOVE', row_idx, col_idx)
                    env.step(action)
                    key = state_key(env)
                    seen_states[key] = seen_states.get(key, 0) + 1
                    
                elif len(move_str) == 3:
                    # ИСПРАВЛЕНИЕ: Стены находятся между клетками.
                    # e1v означает стену между столбцами e(4) и f(3) и строками 1(8) и 2(7).
                    # В нашей матрице это индекс 3 по X и 7 по Y.
                    wall_r = row_idx - 1
                    wall_c = col_idx
                    
                    # Пропускаем ошибочные парсинги (защита от крашей)
                    if wall_r < 0 or wall_r > 7 or wall_c < 0 or wall_c > 7:
                        continue
                        
                    if move_str[2] == 'h':
                        action = move_to_action('WALL_H', wall_r, wall_c)
                    elif move_str[2] == 'v':
                        action = move_to_action('WALL_V', wall_r, wall_c)
                        
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
            
            # Маленький статус-оверлей (чтобы знать, что скрипт жив)
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
                            
                            # Блокируем ходы, которые возвращают в прошлую позицию
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
                            
                            # --- ВНЕДРЕНИЕ ВИЗУАЛЬНОЙ ПОДСВЕТКИ ПРЯМО НА ДОСКУ ---
                            highlight_js = f"""
                            () => {{
                                const svg = document.querySelector('svg[aria-label="Wallz board"]');
                                if (!svg) return;

                                // Удаляем старую подсветку
                                let oldHighlight = document.getElementById('az-visual-hint');
                                if (oldHighlight) oldHighlight.remove();

                                const highlight = document.createElementNS("http://www.w3.org/2000/svg", "g");
                                highlight.id = 'az-visual-hint';
                                highlight.style.pointerEvents = 'none'; // Кликаем сквозь неё

                                const moveType = '{move_type}';
                                const r = {r};
                                const c = {c};

                                let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                                
                                // Формулы на основе SVG-сетки Wallz.gg (ячейки по 60px, отступы по 12px)
                                if (moveType === 'MOVE') {{
                                    rect.setAttribute('x', c * 72);
                                    rect.setAttribute('y', r * 72);
                                    rect.setAttribute('width', 60);
                                    rect.setAttribute('height', 60);
                                    rect.setAttribute('rx', 9);
                                    rect.setAttribute('fill', 'rgba(56, 189, 248, 0.4)'); // Полупрозрачный голубой
                                    rect.setAttribute('stroke', '#38bdf8');
                                    rect.setAttribute('stroke-width', 4);
                                }} else if (moveType === 'WALL_H') {{
                                    rect.setAttribute('x', c * 72);
                                    rect.setAttribute('y', (r * 72) + 60);
                                    rect.setAttribute('width', 132);
                                    rect.setAttribute('height', 12);
                                    rect.setAttribute('rx', 5);
                                    rect.setAttribute('fill', 'rgba(250, 204, 21, 0.8)'); // Желтый
                                    rect.setAttribute('stroke', '#facc15');
                                }} else if (moveType === 'WALL_V') {{
                                    rect.setAttribute('x', (c * 72) + 60);
                                    rect.setAttribute('y', r * 72);
                                    rect.setAttribute('width', 12);
                                    rect.setAttribute('height', 132);
                                    rect.setAttribute('rx', 5);
                                    rect.setAttribute('fill', 'rgba(250, 204, 21, 0.8)'); // Желтый
                                    rect.setAttribute('stroke', '#facc15');
                                }}

                                // Анимация пульсации, чтобы точно заметить периферийным зрением
                                const animate = document.createElementNS("http://www.w3.org/2000/svg", "animate");
                                animate.setAttribute('attributeName', 'opacity');
                                animate.setAttribute('values', '0.3; 1; 0.3');
                                animate.setAttribute('dur', '1s');
                                animate.setAttribute('repeatCount', 'indefinite');
                                rect.appendChild(animate);

                                highlight.appendChild(rect);
                                svg.appendChild(highlight); // Добавляем в конец SVG
                            }}
                            """
                            await page.evaluate(highlight_js)
                            
                            # Обновляем маленький статус
                            await page.evaluate("document.getElementById('az-status').innerText = '👑 Move Highlighted!'")
                            print(f"Move {current_moves + 1} -> Highlighted {move_type} at ({r}, {c})")
                            
                            last_processed_moves = current_moves
                    else:
                        # Убираем подсветку, когда ход противника
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
    
    # Путь к последнему чекпоинту модели
    CHECKPOINT = ROOT_DIR / "wallz_v2" / "checkpoints" / "alphazero_latest.pt"
    
    if not CHECKPOINT.exists():
        print(f"Model not found at {CHECKPOINT}")
        print("Please check the path or run the training script first.")
        sys.exit(1)
        
    assistant = WallzAssistant(CHECKPOINT)
    asyncio.run(assistant.run())