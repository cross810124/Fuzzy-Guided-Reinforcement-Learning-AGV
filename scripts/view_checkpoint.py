#!/usr/bin/env python3
"""
MuJoCo Real-Time Viewer
========================
在 MuJoCo 視窗中即時顯示 AGV 移動。

Usage:
    python scripts/view_sim.py                                          # P-controller
    python scripts/view_sim.py --mode agent --checkpoint data/checkpoints/best.pt   # 看最佳 agent
    python scripts/view_sim.py --mode agent --checkpoint data/checkpoints/latest.pt # 看最新 checkpoint
    python scripts/view_sim.py --mode agent                             # 自動找 best.pt / latest.pt
    python scripts/view_sim.py --mode random                            # 隨機動作
    python scripts/view_sim.py --mode keyboard                          # 鍵盤控制
    python scripts/view_sim.py --mode forward                           # 直線前進
    python scripts/view_sim.py --mode spin                              # 原地旋轉
    python scripts/view_sim.py --speed 0.5                              # 慢動作
    python scripts/view_sim.py --episodes 5                             # 多 episode

鍵盤控制 (Numpad):
    8=前進  2=後退  4=左轉  6=右轉  5=停止  0=重置  9=離開
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.mujoco.navigation_env import NavigationEnv


# ================================================================
# Keyboard state & callback
# ================================================================

key_state = {
    "forward": False, "backward": False,
    "left": False, "right": False,
    "reset": False, "quit": False,
}


def key_callback(keycode):
    if keycode == 328:       key_state["forward"] = True; key_state["backward"] = False
    elif keycode == 322:     key_state["backward"] = True; key_state["forward"] = False
    elif keycode == 324:     key_state["left"] = True; key_state["right"] = False
    elif keycode == 326:     key_state["right"] = True; key_state["left"] = False
    elif keycode == 325:     key_state["forward"] = key_state["backward"] = key_state["left"] = key_state["right"] = False
    elif keycode == 320:     key_state["reset"] = True
    elif keycode == 329:     key_state["quit"] = True


def get_keyboard_action() -> np.ndarray:
    left = right = 0.0
    if key_state["forward"]:  left += 0.6; right += 0.6
    if key_state["backward"]: left -= 0.6; right -= 0.6
    if key_state["left"]:     left -= 0.4; right += 0.4
    if key_state["right"]:    left += 0.4; right -= 0.4
    return np.array([np.clip(left, -1, 1), np.clip(right, -1, 1)], dtype=np.float32)


def get_navigate_action(obs, goal_pos):
    goal_angle = obs[8]
    goal_dist = np.linalg.norm(obs[:2] - goal_pos)
    base_speed = np.clip(goal_dist * 0.8, 0.1, 0.7)
    turn = np.clip(goal_angle * 1.5, -0.8, 0.8)
    return np.array([np.clip(base_speed + turn, -1, 1),
                     np.clip(base_speed - turn, -1, 1)], dtype=np.float32)


# ================================================================
# Agent loader
# ================================================================

def load_agent(checkpoint_path: str):
    """載入訓練好的 PPO Agent。"""
    from agent.ppo_agent import PPOAgent, PPOConfig

    print(f"  Loading checkpoint: {checkpoint_path}")
    agent = PPOAgent(PPOConfig())
    agent.load(checkpoint_path)
    agent.network.eval()
    print(f"  Agent loaded! (steps={agent.total_steps:,}, "
          f"episodes={agent.total_episodes:,}, "
          f"updates={agent.update_count})")
    return agent


# ================================================================
# Main viewer
# ================================================================

def run_viewer(mode: str, speed: float, episodes: int, seed: int,
               checkpoint: str = None):
    env = NavigationEnv()
    dt = env.model.opt.timestep * env.sim_steps_per_action

    # Load agent if needed
    agent = None
    if mode == "agent":
        if not checkpoint:
            for default_path in ["data/checkpoints/best.pt",
                                 "data/checkpoints/latest.pt"]:
                full_path = PROJECT_ROOT / default_path
                if full_path.exists():
                    checkpoint = str(full_path)
                    break
            if not checkpoint:
                print("  ERROR: No checkpoint found!")
                print("  Usage: python scripts/view_sim.py --mode agent --checkpoint <path>")
                return
        agent = load_agent(checkpoint)

    print(f"\n  Mode:     {mode}")
    print(f"  Speed:    {speed}x")
    print(f"  Episodes: {episodes}")

    if mode == "keyboard":
        print("\n  ╔════════════════════════════════════╗")
        print("  ║  8 = 前進      2 = 後退           ║")
        print("  ║  4 = 左轉      6 = 右轉           ║")
        print("  ║  5 = 停止      0 = 重置           ║")
        print("  ║  9 = 離開                          ║")
        print("  ║  按一下啟動，5 停止                ║")
        print("  ╚════════════════════════════════════╝")

    if mode == "agent":
        print("\n  ╔════════════════════════════════════════════╗")
        print("  ║  Watching trained agent (deterministic)   ║")
        print("  ║  Numpad 0 = 重置    Numpad 9 = 離開       ║")
        print("  ╚════════════════════════════════════════════╝")

    print("\n  Starting viewer...\n")

    # Stats
    total_successes = 0
    total_collisions = 0
    total_timeouts = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        goal_pos = info["goal_pos"].copy()
        start_dist = np.linalg.norm(obs[:2] - goal_pos)

        print(f"  ── Episode {ep+1}/{episodes} ──")
        print(f"     Start: ({obs[0]:+.1f}, {obs[1]:+.1f})  "
              f"Goal: ({goal_pos[0]:+.1f}, {goal_pos[1]:+.1f})  "
              f"Dist: {start_dist:.1f}m")

        for k in key_state:
            key_state[k] = False

        step = 0
        total_reward = 0.0
        running = True

        with mujoco.viewer.launch_passive(
            env.model, env.data,
            key_callback=key_callback,
        ) as viewer:
            viewer.cam.distance = 12.0
            viewer.cam.elevation = -55.0
            viewer.cam.azimuth = 90.0
            viewer.cam.lookat[:] = [0, 0, 0]

            while viewer.is_running() and running:
                step_start = time.time()

                if key_state["quit"]:
                    running = False
                    break
                if key_state["reset"]:
                    key_state["reset"] = False
                    print("     [RESET]")
                    obs, info = env.reset(seed=seed + ep)
                    goal_pos = info["goal_pos"].copy()
                    step = 0
                    total_reward = 0.0
                    viewer.sync()
                    continue

                # ---- Get action ----
                guidance = None
                if mode == "agent":
                    action, guidance, action_info = agent.select_action(
                        obs, deterministic=True)
                elif mode == "keyboard":
                    action = get_keyboard_action()
                elif mode == "navigate":
                    action = get_navigate_action(obs, goal_pos)
                elif mode == "forward":
                    action = np.array([0.6, 0.6], dtype=np.float32)
                elif mode == "spin":
                    action = np.array([0.6, -0.6], dtype=np.float32)
                else:
                    action = env.action_space.sample()

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                viewer.sync()

                # Status print
                if step % 50 == 0:
                    dist = info["reward_info"]["goal_dist"]
                    act_str = f"L={action[0]:+.2f} R={action[1]:+.2f}"
                    extra = ""
                    if mode == "agent" and guidance is not None:
                        extra = (f" | safe={guidance.safety_level:.2f}"
                                 f" expl={guidance.exploration_encouragement:.2f}"
                                 f" conf={guidance.action_confidence:.2f}"
                                 f" spd={guidance.speed_modifier:.2f}")
                    print(f"     Step {step:3d} | pos=({obs[0]:+5.1f}, {obs[1]:+5.1f}) | "
                          f"dist={dist:.2f}m | {act_str} | Σr={total_reward:+.1f}{extra}")

                # Episode end
                if terminated or truncated:
                    success = info.get("success", False)
                    collision = info.get("collision", False)

                    if success:
                        result = "✓ SUCCESS"
                        total_successes += 1
                    elif collision:
                        result = "✗ COLLISION"
                        total_collisions += 1
                    else:
                        result = "⏱ TIMEOUT"
                        total_timeouts += 1

                    print(f"     >>> {result} | {step} steps | Σreward={total_reward:+.1f}")

                    if mode == "keyboard":
                        print("     Press 0 to reset, 9 to quit")
                        while viewer.is_running():
                            viewer.sync()
                            time.sleep(0.05)
                            if key_state["reset"]:
                                key_state["reset"] = False
                                obs, info = env.reset(seed=seed + ep + 1000)
                                goal_pos = info["goal_pos"].copy()
                                step = 0
                                total_reward = 0.0
                                print(f"\n     [RESET] Goal: ({goal_pos[0]:+.1f}, {goal_pos[1]:+.1f})")
                                break
                            if key_state["quit"]:
                                running = False
                                break
                        if not running:
                            break
                        continue
                    else:
                        if ep < episodes - 1:
                            print(f"     Next episode in 2s...")
                            pause_end = time.time() + 2.0
                            while time.time() < pause_end and viewer.is_running():
                                viewer.sync()
                                time.sleep(0.05)
                        break

                # Pacing
                elapsed = time.time() - step_start
                sleep_time = (dt / speed) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if key_state["quit"] or not running:
            break

    # Print summary for agent mode
    total_eps = total_successes + total_collisions + total_timeouts
    if total_eps > 0 and mode == "agent":
        print(f"\n  ═══════════════════════════════════")
        print(f"  Agent Summary ({total_eps} episodes):")
        print(f"    Success:   {total_successes}/{total_eps} ({total_successes/total_eps:.0%})")
        print(f"    Collision: {total_collisions}/{total_eps} ({total_collisions/total_eps:.0%})")
        print(f"    Timeout:   {total_timeouts}/{total_eps} ({total_timeouts/total_eps:.0%})")
        print(f"  ═══════════════════════════════════")

    env.close()
    print("\n  Viewer closed.")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Real-Time Viewer")
    parser.add_argument("--mode", type=str, default="navigate",
                        choices=["navigate", "random", "forward", "spin",
                                 "keyboard", "agent"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to agent checkpoint (for --mode agent)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (0.5=slow, 2.0=fast)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 50)
    print("  Fuzzy-Guided RL AGV — MuJoCo Viewer")
    print("=" * 50)

    run_viewer(args.mode, args.speed, args.episodes, args.seed,
               checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
