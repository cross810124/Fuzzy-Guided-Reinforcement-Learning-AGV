#!/usr/bin/env python3
"""
MuJoCo Real-Time Viewer
========================
在 MuJoCo 視窗中即時顯示 AGV 移動。

Usage:
    python scripts/view_sim.py                     # P-controller 導航
    python scripts/view_sim.py --mode random       # 隨機動作
    python scripts/view_sim.py --mode forward      # 直線前進
    python scripts/view_sim.py --mode spin         # 原地旋轉
    python scripts/view_sim.py --mode keyboard     # 鍵盤控制
    python scripts/view_sim.py --speed 0.5         # 慢動作 (0.5x)
    python scripts/view_sim.py --episodes 3        # 連跑 3 個 episode

需要有顯示器 (display)，headless 環境無法使用。

鍵盤控制模式 (--mode keyboard):
    8      : 前進
    2      : 後退
    4      : 左轉
    6      : 右轉
    5      : 停止
    0      : 重置 episode
    9      : 離開
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
# Keyboard state (module-level so the callback can access it)
# ================================================================
key_state = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False,
    "reset": False,
    "quit": False,
}


def key_callback(keycode):
    """
    MuJoCo passive viewer key callback.
    
    Numpad keycodes (GLFW):
        KP_0=320, KP_2=322, KP_4=324, KP_5=325, KP_6=326, KP_8=328, KP_9=329
    
    Controls:
        8 = 前進, 2 = 後退, 4 = 左轉, 6 = 右轉
        5 = 停止, 0 = 重置, 9 = 離開
    """
    if keycode == 328:       # KP_8 - forward
        key_state["forward"] = True
        key_state["backward"] = False
    elif keycode == 322:     # KP_2 - backward
        key_state["backward"] = True
        key_state["forward"] = False
    elif keycode == 324:     # KP_4 - left
        key_state["left"] = True
        key_state["right"] = False
    elif keycode == 326:     # KP_6 - right
        key_state["right"] = True
        key_state["left"] = False
    elif keycode == 325:     # KP_5 - stop all
        key_state["forward"] = False
        key_state["backward"] = False
        key_state["left"] = False
        key_state["right"] = False
    elif keycode == 320:     # KP_0 - reset
        key_state["reset"] = True
    elif keycode == 329:     # KP_9 - quit
        key_state["quit"] = True


def get_keyboard_action() -> np.ndarray:
    """Convert key_state to wheel velocities."""
    left = 0.0
    right = 0.0
    speed = 0.6
    turn_speed = 0.4

    if key_state["forward"]:
        left += speed
        right += speed
    if key_state["backward"]:
        left -= speed
        right -= speed
    if key_state["left"]:
        left -= turn_speed
        right += turn_speed
    if key_state["right"]:
        left += turn_speed
        right -= turn_speed

    return np.array([np.clip(left, -1, 1), np.clip(right, -1, 1)], dtype=np.float32)


def get_navigate_action(obs: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """Simple P-controller navigation."""
    goal_angle = obs[8]
    goal_dist = np.linalg.norm(obs[:2] - goal_pos)
    base_speed = np.clip(goal_dist * 0.8, 0.1, 0.7)
    turn = np.clip(goal_angle * 1.5, -0.8, 0.8)
    return np.array([
        np.clip(base_speed + turn, -1, 1),
        np.clip(base_speed - turn, -1, 1),
    ], dtype=np.float32)


def run_viewer(mode: str, speed: float, episodes: int, seed: int):
    """Main viewer loop."""
    env = NavigationEnv()
    dt = env.model.opt.timestep * env.sim_steps_per_action

    print(f"\n  Mode:     {mode}")
    print(f"  Speed:    {speed}x")
    print(f"  Episodes: {episodes}")
    if mode == "keyboard":
        print("\n  ╔════════════════════════════════════╗")
        print("  ║  8 = 前進      2 = 後退           ║")
        print("  ║  4 = 左轉      6 = 右轉           ║")
        print("  ║  5 = 停止      0 = 重置           ║")
        print("  ║  9 = 離開                          ║")
        print("  ║                                    ║")
        print("  ║  按一下啟動，5 停止                ║")
        print("  ╚════════════════════════════════════╝")
    print("\n  Starting viewer...\n")

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        goal_pos = info["goal_pos"].copy()
        start_dist = np.linalg.norm(obs[:2] - goal_pos)

        print(f"  ── Episode {ep+1}/{episodes} ──")
        print(f"     Start: ({obs[0]:+.1f}, {obs[1]:+.1f})  "
              f"Goal: ({goal_pos[0]:+.1f}, {goal_pos[1]:+.1f})  "
              f"Dist: {start_dist:.1f}m")

        # Reset key state
        for k in key_state:
            key_state[k] = False

        step = 0
        total_reward = 0.0
        running = True

        with mujoco.viewer.launch_passive(
            env.model, env.data,
            key_callback=key_callback,
        ) as viewer:
            # Camera: top-down angled view
            viewer.cam.distance = 12.0
            viewer.cam.elevation = -55.0
            viewer.cam.azimuth = 90.0
            viewer.cam.lookat[:] = [0, 0, 0]

            while viewer.is_running() and running:
                step_start = time.time()

                # Check reset/quit
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

                # Get action
                if mode == "keyboard":
                    action = get_keyboard_action()
                elif mode == "navigate":
                    action = get_navigate_action(obs, goal_pos)
                elif mode == "forward":
                    action = np.array([0.6, 0.6], dtype=np.float32)
                elif mode == "spin":
                    action = np.array([0.6, -0.6], dtype=np.float32)
                else:  # random
                    action = env.action_space.sample()

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                # Sync viewer
                viewer.sync()

                # Status print
                if step % 50 == 0:
                    dist = info["reward_info"]["goal_dist"]
                    act_str = f"L={action[0]:+.2f} R={action[1]:+.2f}"
                    print(f"     Step {step:3d} | pos=({obs[0]:+5.1f}, {obs[1]:+5.1f}) | "
                          f"dist={dist:.2f}m | {act_str} | Σr={total_reward:+.1f}")

                # Episode end
                if terminated or truncated:
                    result = ("✓ SUCCESS" if info.get("success")
                              else "✗ COLLISION" if info.get("collision")
                              else "⏱ TIMEOUT")
                    print(f"     >>> {result} | {step} steps | Σreward={total_reward:+.1f}")

                    if mode == "keyboard":
                        print("     Press 0 to reset, 9 to quit")
                        # Wait for user input in keyboard mode
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
                        # Auto mode: pause 2s then next episode
                        if ep < episodes - 1:
                            print(f"     Next episode in 2s...")
                            pause_end = time.time() + 2.0
                            while time.time() < pause_end and viewer.is_running():
                                viewer.sync()
                                time.sleep(0.05)
                        break

                # Real-time pacing
                elapsed = time.time() - step_start
                sleep_time = (dt / speed) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if key_state["quit"] or not running:
            break

    env.close()
    print("\n  Viewer closed.")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Real-Time Viewer")
    parser.add_argument("--mode", type=str, default="navigate",
                        choices=["navigate", "random", "forward", "spin", "keyboard"])
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (0.5=slow, 2.0=fast)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 50)
    print("  Fuzzy-Guided RL AGV — MuJoCo Viewer")
    print("=" * 50)

    run_viewer(args.mode, args.speed, args.episodes, args.seed)


if __name__ == "__main__":
    main()