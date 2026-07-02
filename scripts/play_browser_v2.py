import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from sb3_contrib import MaskablePPO

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / "wallz_v2"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.env.action_space import NUM_SQUARES, TOTAL_ACTIONS, action_to_move
from wallz_v2.env.gym_env import WallzGymEnv
from wallz_v2.env.wallz_env import WallzEnv

PROFILE_DIR = PACKAGE_DIR / "browser_profile"
WALLZ_URL = "https://wallz.gg/"
DEFAULT_MODEL = PACKAGE_DIR / "checkpoints" / "ppo_v2_model.zip"

RGB_RE = re.compile(r"rgba?\((\d+),\s*(\d+),\s*(\d+)")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def resolve_model_path(value: str) -> Path:
    if value in {"ppo", "best", "default"}:
        return DEFAULT_MODEL
    path = Path(value).expanduser()
    return path if path.is_absolute() else PACKAGE_DIR / path


def action_name(action: int) -> str:
    if action < 0:
        return "NO_MODEL"
    move_type, (r, c) = action_to_move(int(action))
    if move_type == "MOVE":
        return f"MOVE_{r}_{c}"
    return f"{move_type}_{r}_{c}"


def rgb(value: str):
    match = RGB_RE.search(value or "")
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def color_key(item: dict) -> str:
    text = f"{item.get('fill', '')} {item.get('stroke', '')} {item.get('className', '')}".lower()
    if any(token in text for token in ("cyan", "teal", "turquoise", "emerald")):
        return "teal"
    if any(token in text for token in ("pink", "rose", "red", "crimson", "salmon")):
        return "red"

    for key in ("fill", "stroke"):
        value = rgb(item.get(key, ""))
        if value is None:
            continue
        r, g, b = value
        if g >= r + 15 and b >= r + 15:
            return "teal"
        if r >= g + 15 and r >= b - 10:
            return "red"

    return "unknown"


def cluster_axis(values, expected=9):
    values = sorted(v for v in values if np.isfinite(v))
    if not values:
        return []
    if len(values) <= expected:
        return values

    span = values[-1] - values[0]
    threshold = max(8.0, span / 80.0)

    clusters = []
    current = [values[0]]
    for value in values[1:]:
        if abs(value - current[-1]) <= threshold:
            current.append(value)
        else:
            clusters.append(current)
            current = [value]
    clusters.append(current)

    centers = [sum(group) / len(group) for group in clusters]
    if len(centers) > expected:
        centers = sorted(centers, key=lambda c: sum(abs(v - c) for v in values))[:expected]

    return sorted(centers)


def nearest_index(value, centers):
    return min(range(len(centers)), key=lambda i: abs(value - centers[i]))


class BrowserAgentV2:
    def __init__(self, model_path: Path, max_turn_seconds: float, allow_backward: bool):
        self.env = WallzEnv()
        self.gym_env = WallzGymEnv()
        self.obs, self.mask = self.env.reset()

        self.model_path = model_path
        self.max_turn_seconds = float(max_turn_seconds)
        self.allow_backward = bool(allow_backward)

        self.own_color = None
        self.last_own_xy = None
        self.last_centers = None
        self.position_history = []
        self.walls_left = 10
        self.model = None

        if model_path.exists():
            self.model = MaskablePPO.load(
                str(model_path),
                env=self.gym_env,
                device="cpu",
                custom_objects={
                    "observation_space": self.gym_env.observation_space,
                    "action_space": self.gym_env.action_space,
                },
            )
            print(f"[System] Model loaded: {model_path}")
        else:
            print(f"[System] Model not found: {model_path}; using BFS fallback policy")

    def run(self, url: str):
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                headless=False,
                slow_mo=60,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(url, wait_until="domcontentloaded")

            input("\n[Control] Start a match and press ENTER when the board is visible...")

            try:
                page.locator('svg[aria-label="Wallz board"]').first.wait_for(state="visible", timeout=40_000)
            except PlaywrightTimeoutError:
                print("[Error] Board SVG not found")
                context.close()
                return

            try:
                self.play_loop(page)
            finally:
                context.close()

    def read_board(self, page):
        return page.evaluate(
            """
            () => {
                const svg = document.querySelector('svg[aria-label="Wallz board"]');
                if (!svg) return null;

                function num(node, name) {
                    const value = node.getAttribute(name);
                    if (value === null) return null;
                    const parsed = Number(value);
                    return Number.isFinite(parsed) ? parsed : null;
                }

                function read(node) {
                    const rect = node.getBoundingClientRect();
                    const style = window.getComputedStyle(node);
                    return {
                        tag: node.tagName.toLowerCase(),
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        w: rect.width,
                        h: rect.height,
                        r: Math.max(rect.width, rect.height) / 2,
                        rawX: num(node, 'x'),
                        rawY: num(node, 'y'),
                        rawW: num(node, 'width'),
                        rawH: num(node, 'height'),
                        rawCx: num(node, 'cx'),
                        rawCy: num(node, 'cy'),
                        rawR: num(node, 'r'),
                        fill: style.fill || node.getAttribute('fill') || '',
                        attrFill: node.getAttribute('fill') || '',
                        stroke: style.stroke || node.getAttribute('stroke') || '',
                        opacity: Number(style.opacity || 1),
                        className: node.getAttribute('class') || '',
                    };
                }

                const items = Array.from(svg.querySelectorAll('rect,circle'))
                    .map(read)
                    .filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y) && item.opacity > 0.03);

                return {
                    circles: items.filter((item) => item.tag === 'circle'),
                    rects: items.filter((item) => item.tag === 'rect'),
                };
            }
            """
        )

    def cell_centers(self, state):
        cells = []
        for item in state["rects"]:
            raw_cell = item.get("rawW") == 60 and item.get("rawH") == 60
            ratio = item["w"] / item["h"] if item["h"] else 0
            screen_cell = 0.65 <= ratio <= 1.55 and 25 <= item["w"] <= 140 and 25 <= item["h"] <= 140
            if raw_cell or screen_cell:
                cells.append(item)

        xs = cluster_axis([item["x"] for item in cells], expected=9)
        ys = cluster_axis([item["y"] for item in cells], expected=9)
        return xs[:9], ys[:9]

    def grid_pos(self, item, centers):
        xs, ys = centers
        return nearest_index(item["x"], xs), nearest_index(item["y"], ys)

    def parse_walls(self, state, centers):
        horizontal = set()
        vertical = set()

        xs, ys = centers
        bx = [(xs[i] + xs[i + 1]) / 2.0 for i in range(8)] if len(xs) >= 9 else []
        by = [(ys[i] + ys[i + 1]) / 2.0 for i in range(8)] if len(ys) >= 9 else []

        for item in state["rects"]:
            raw_w = item.get("rawW")
            raw_h = item.get("rawH")
            ratio = item["w"] / item["h"] if item["h"] else 0

            raw_h_wall = raw_w is not None and raw_h is not None and abs(raw_w - 132) <= 18 and abs(raw_h - 12) <= 8
            raw_v_wall = raw_w is not None and raw_h is not None and abs(raw_w - 12) <= 8 and abs(raw_h - 132) <= 18
            screen_h_wall = ratio >= 3.0 and item["w"] >= 30 and item["h"] <= 24
            screen_v_wall = ratio <= 0.33 and item["h"] >= 30 and item["w"] <= 24

            is_h = raw_h_wall or screen_h_wall
            is_v = raw_v_wall or screen_v_wall
            if not is_h and not is_v:
                continue

            text = f"{item.get('attrFill', '')} {item.get('fill', '')}".lower()
            if "color-p" not in text:
                continue

            if len(bx) < 8 or len(by) < 8:
                continue

            c = nearest_index(item["x"], bx)
            r = nearest_index(item["y"], by)

            if is_h:
                horizontal.add((r, c))
            elif is_v:
                vertical.add((r, c))

        return horizontal, vertical

    def pick_pawns(self, state):
        circles = state["circles"]
        if not circles:
            return None, None

        max_radius = max(item["r"] for item in circles)
        pawns = []
        for item in circles:
            fill_text = f"{item.get('fill', '')} {item.get('stroke', '')}".lower()
            if item["r"] >= max(10.0, max_radius * 0.65) and "none" not in fill_text:
                pawns.append(item)

        if len(pawns) < 2:
            pawns = sorted(circles, key=lambda item: item["r"], reverse=True)[:2]
        if not pawns:
            return None, None

        def dist_last(item):
            if self.last_own_xy is None:
                return 0.0
            return abs(item["x"] - self.last_own_xy[0]) + abs(item["y"] - self.last_own_xy[1])

        if self.own_color is None:
            own = max(pawns, key=lambda item: (item["y"], item["r"]))
            self.own_color = color_key(own)
            print(f"[Vision] Bound own pawn color={self.own_color}")
        else:
            same_color = [item for item in pawns if color_key(item) == self.own_color]
            own = min(same_color, key=dist_last) if same_color else min(pawns, key=dist_last)

        self.last_own_xy = (own["x"], own["y"])
        opponents = [item for item in pawns if item is not own]
        opponent = max(opponents, key=dist_last) if opponents else None
        return own, opponent

    def read_walls_left(self, page):
        try:
            value = page.evaluate(
                """
                () => {
                    const text = document.body.innerText || '';
                    const lower = text.toLowerCase();
                    const youIndex = lower.indexOf('you');
                    if (youIndex >= 0) {
                        const chunk = text.slice(youIndex, youIndex + 350);
                        const match = chunk.match(/WALLS\s*[·:.\-]?\s*(\d+)/i);
                        if (match) return Number(match[1]);
                    }
                    const matches = [...text.matchAll(/WALLS\s*[·:.\-]?\s*(\d+)/gi)].map((m) => Number(m[1]));
                    return matches.length ? matches[matches.length - 1] : null;
                }
                """
            )
        except Exception:
            return

        if isinstance(value, (int, float)) and np.isfinite(value):
            self.walls_left = int(max(0, min(10, value)))
            self.env.walls_left[1] = self.walls_left

    def read_timer_seconds(self, page):
        try:
            text = page.evaluate("() => document.body.innerText || ''")
        except Exception:
            return None

        matches = TIME_RE.findall(text or "")
        if not matches:
            return None

        minutes, seconds = matches[0]
        return int(minutes) * 60 + int(seconds)

    def sync_env(self, page, own, opponent, state, centers):
        self.last_centers = centers

        p1_pos = self.grid_pos(own, centers)
        p2_pos = self.grid_pos(opponent, centers) if opponent else self.env.p2_pos
        horizontal, vertical = self.parse_walls(state, centers)

        self.env.h_walls[:, :] = False
        self.env.v_walls[:, :] = False

        for r, c in horizontal:
            if 0 <= r < 8 and 0 <= c < 8:
                self.env.h_walls[r, c] = True

        for r, c in vertical:
            if 0 <= r < 8 and 0 <= c < 8:
                self.env.v_walls[r, c] = True

        self.env.p1_pos = p1_pos
        self.env.p2_pos = p2_pos
        self.env.current_player = 1
        self.read_walls_left(page)

        self.obs = self.env.get_observation()
        self.mask = self.env.get_legal_action_mask()
        return p1_pos, p2_pos

    def move_options(self, own, state, centers):
        own_radius = own["r"]
        legal_mask = self.env.get_legal_action_mask()

        options = {}
        for circle in state["circles"]:
            if circle is own or circle["r"] >= own_radius * 0.75:
                continue

            target_pos = self.grid_pos(circle, centers)
            x, y = target_pos
            action = y * 9 + x

            if 0 <= action < NUM_SQUARES and legal_mask[action]:
                options[action] = {
                    "kind": "move",
                    "x": circle["x"],
                    "y": circle["y"],
                    "target_pos": target_pos,
                }

        return options

    def choose_action(self, predicted, options, p1_pos, force_fast=False, force_move_only=False):
        move_actions = [a for a in options if a < NUM_SQUARES]
        fast = bool(force_fast or force_move_only)

        def move_score(action):
            guard = f" guard {action_name(predicted)}->{action_name(action)}" if overridden and predicted >= 0 else ""

            if action < 81:

                target = options[action]

                target_pos = target["target_pos"]

                print(f"[Move] P1={p1_pos} P2={p2_pos} {action_name(action)}->{target_pos}{guard}")

                page.mouse.click(target["x"], target["y"])

            else:

                from wallz_v2.env.action_space import action_to_move

                move_type, (r, c) = action_to_move(action)

                xs, ys = centers

                click_x = (xs[c] + xs[c + 1]) / 2.0

                click_y = (ys[r] + ys[r + 1]) / 2.0

                print(f"[{move_type}] P1={p1_pos} P2={p2_pos} r={r} c={c}{guard}")

                page.mouse.click(click_x, click_y)

                self.position_history.append(target_pos)
                self.position_history = self.position_history[-12:]
                time.sleep(1.0)

            except KeyboardInterrupt:
                print("\n[System] stopped")
                break
            except Exception as exc:
                print(f"[Error] {type(exc).__name__}: {exc}")
                time.sleep(1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Wallz v2 browser agent, move-only stable mode")
    parser.add_argument("--model", default="ppo", help="Model alias/path. Default: ppo -> checkpoints/ppo_v2_model.zip")
    parser.add_argument("--url", default=WALLZ_URL)
    parser.add_argument("--max-turn-seconds", type=float, default=8.0, help="Local per-turn guard; force BFS move if exceeded")
    parser.add_argument("--allow-backward", action="store_true", help="Allow backward pawn moves. Default blocks them when alternatives exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    agent = BrowserAgentV2(
        model_path=resolve_model_path(args.model),
        max_turn_seconds=args.max_turn_seconds,
        allow_backward=args.allow_backward,
    )
    print("[System] Browser agent mode: no ProgressGuard")
    print(f"[System] Browser profile: {PROFILE_DIR}")
    agent.run(args.url)


if __name__ == "__main__":
    main()
