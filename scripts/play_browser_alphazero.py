import argparse
import copy
import inspect
import sys
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT_DIR / "wallz_v2"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wallz_v2.agents.model import WallzNet
from wallz_v2.agents.mcts import MCTS
from play_browser_v2 import BrowserAgentV2, WALLZ_URL


class AlphaZeroPredictor:
    def __init__(self, checkpoint: Path, device: str = "cpu", mcts_simulations: int = 8):
        self.device = torch.device(device)
        self.model = WallzNet().to(self.device)

        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        self.model.load_state_dict(state)
        self.model.eval()

        self.mcts_simulations = int(mcts_simulations)
        self.mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
        self.state_history = deque(maxlen=12)
        self.p1_position_history = deque(maxlen=8)

        print(f"[AlphaZero] Loaded: {checkpoint}")
        print(f"[AlphaZero] MCTS simulations: {self.mcts_simulations}")

    def action_repeats_own_position(self, env, action):
        sim_env = copy.deepcopy(env)
        sim_env.step(int(action))

        # For browser inference we care about our pawn oscillating,
        # even if opponent position changed meanwhile.
        if sim_env.p1_pos in list(self.p1_position_history)[-4:]:
            return True

        return False



    def raw_predict(self, obs, action_masks=None):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.ndim == 3:
                obs_t = obs_t.unsqueeze(0)

            mask_t = None
            if action_masks is not None:
                mask_np = np.asarray(action_masks, dtype=bool)
                if mask_np.ndim == 1:
                    mask_np = mask_np[None, :]
                mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)

            logits, _ = self.model(obs_t, mask_t)
            return int(torch.argmax(logits, dim=1).item())

    def predict_from_env(self, env, obs, action_masks):
        mask = np.asarray(action_masks, dtype=bool)

        current_key = self.state_key(env)
        if not self.state_history or self.state_history[-1] != current_key:
            self.state_history.append(current_key)

        if not self.p1_position_history or self.p1_position_history[-1] != env.p1_pos:
            self.p1_position_history.append(env.p1_pos)

        try:
            probs = self.mcts.get_action_prob(env, temperature=1.0)
            probs = np.asarray(probs, dtype=np.float64)
            probs[~mask] = 0.0

            if probs.sum() > 0:
                ranked = np.argsort(probs)[::-1]
                best_repeat = None

                for action in ranked:
                    action = int(action)
                    if probs[action] <= 0:
                        break

                    if self.action_repeats_state(env, action) or self.action_repeats_own_position(env, action):
                        if best_repeat is None:
                            best_repeat = action
                        continue

                    print(f"[AlphaZeroMCTS] action={action} prob={probs[action]:.3f}")
                    return action


                if best_repeat is not None:
                    print(f"[AlphaZeroMCTS] no escape, least-bad repeat action={best_repeat} prob={probs[best_repeat]:.3f}")
                    return best_repeat

        except Exception as exc:
            print(f"[AlphaZeroMCTS] fallback raw policy: {type(exc).__name__}: {exc}")

        action = self.raw_predict(obs, action_masks=mask)
        print(f"[AlphaZeroRaw] action={action}")
        return action


def resolve_checkpoint(value: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates += [PACKAGE_DIR / raw, ROOT_DIR / raw]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Checkpoint not found. Tried: " + ", ".join(str(p) for p in candidates))


def make_browser_agent(args):
    dummy_ppo_path = PACKAGE_DIR / "__alphazero_no_ppo__.zip"

    sig = inspect.signature(BrowserAgentV2)
    kwargs = {}

    if "model_path" in sig.parameters:
        kwargs["model_path"] = dummy_ppo_path
    if "max_turn_seconds" in sig.parameters:
        kwargs["max_turn_seconds"] = args.max_turn_seconds
    if "allow_backward" in sig.parameters:
        kwargs["allow_backward"] = True
    if "allow_walls" in sig.parameters:
        kwargs["allow_walls"] = not args.no_walls
    if "wall_fail_limit" in sig.parameters:
        kwargs["wall_fail_limit"] = args.wall_fail_limit

    return BrowserAgentV2(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaZero checkpoint in Wallz browser agent with MCTS")
    parser.add_argument("--checkpoint", default="checkpoints/alphazero_no_loop_epoch_9.pt")
    parser.add_argument("--url", default=WALLZ_URL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts-simulations", type=int, default=8)
    parser.add_argument("--no-walls", action="store_true")
    parser.add_argument("--wall-fail-limit", type=int, default=2)
    parser.add_argument("--max-turn-seconds", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)

    agent = make_browser_agent(args)
    predictor = AlphaZeroPredictor(
        checkpoint,
        device=args.device,
        mcts_simulations=args.mcts_simulations,
    )

    agent.model = predictor
    agent.model_path = checkpoint

    def predict_action_with_mcts(self, masks):
        full_legal_mask = self.env.get_legal_action_mask()
        return predictor.predict_from_env(self.env, self.obs, full_legal_mask)

    agent.predict_action = types.MethodType(predict_action_with_mcts, agent)

    mode = "AlphaZero + MCTS full control" if not args.no_walls else "AlphaZero + MCTS move-only"
    print(f"[System] Browser agent mode: {mode}")
    agent.run(args.url)


if __name__ == "__main__":
    main()
