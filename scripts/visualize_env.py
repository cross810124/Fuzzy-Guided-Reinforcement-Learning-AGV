#!/usr/bin/env python3
"""
Environment Visualization Script
==================================
視覺化 NavigationEnv 的測試結果。

Usage:
    python scripts/visualize_env.py                    # 預設：P-controller 導航
    python scripts/visualize_env.py --mode random      # 隨機動作軌跡
    python scripts/visualize_env.py --mode forward     # 直線前進
    python scripts/visualize_env.py --mode spin        # 原地旋轉
    python scripts/visualize_env.py --mode multi       # 多 episode 比較
    python scripts/visualize_env.py --episodes 10      # 多 episode
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.mujoco.navigation_env import NavigationEnv

OUTPUT_DIR = PROJECT_ROOT / "data" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Color palette ---
COLORS = {
    "robot": "#2563EB",
    "goal": "#16A34A",
    "wall": "#6B7280",
    "trajectory": "#3B82F6",
    "start": "#EF4444",
    "lidar": "#F59E0B",
    "reward_pos": "#22C55E",
    "reward_neg": "#EF4444",
}


def run_episode(env, controller="navigate", seed=42):
    """Run one episode and record all data."""
    obs, info = env.reset(seed=seed)
    goal_pos = info["goal_pos"].copy()

    history = {
        "obs": [obs.copy()],
        "actions": [],
        "rewards": [],
        "reward_info": [],
        "positions": [obs[:2].copy()],
        "yaws": [obs[2]],
        "lidar": [obs[7]],
        "goal_angles": [obs[8]],
        "linear_vels": [obs[3]],
        "angular_vels": [obs[4]],
    }

    for step in range(env.max_steps):
        if controller == "navigate":
            goal_angle = obs[8]
            goal_dist = np.linalg.norm(obs[:2] - goal_pos)
            base_speed = np.clip(goal_dist * 0.8, 0.1, 0.7)
            turn = np.clip(goal_angle * 1.5, -0.8, 0.8)
            action = np.array([
                np.clip(base_speed + turn, -1, 1),
                np.clip(base_speed - turn, -1, 1),
            ], dtype=np.float32)
        elif controller == "forward":
            action = np.array([0.8, 0.8], dtype=np.float32)
        elif controller == "spin":
            action = np.array([0.8, -0.8], dtype=np.float32)
        else:  # random
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        history["obs"].append(obs.copy())
        history["actions"].append(action.copy())
        history["rewards"].append(reward)
        history["reward_info"].append(info.get("reward_info", {}))
        history["positions"].append(obs[:2].copy())
        history["yaws"].append(obs[2])
        history["lidar"].append(obs[7])
        history["goal_angles"].append(obs[8])
        history["linear_vels"].append(obs[3])
        history["angular_vels"].append(obs[4])

        if terminated or truncated:
            history["result"] = (
                "SUCCESS" if info.get("success")
                else "COLLISION" if info.get("collision")
                else "TIMEOUT"
            )
            break

    history["result"] = history.get("result", "TIMEOUT")
    history["goal_pos"] = goal_pos
    history["steps"] = len(history["actions"])

    # Convert to arrays
    for key in ["positions", "rewards", "lidar", "goal_angles",
                "linear_vels", "angular_vels", "yaws"]:
        history[key] = np.array(history[key])

    return history


def draw_arena(ax, arena_size=5.0):
    """Draw the arena walls and grid."""
    wall = patches.Rectangle(
        (-arena_size, -arena_size), arena_size * 2, arena_size * 2,
        linewidth=2, edgecolor=COLORS["wall"], facecolor="none",
        linestyle="-", zorder=1,
    )
    ax.add_patch(wall)
    ax.set_xlim(-arena_size - 0.5, arena_size + 0.5)
    ax.set_ylim(-arena_size - 0.5, arena_size + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_facecolor("#FAFAFA")


def draw_robot(ax, pos, yaw, size=0.2, color=COLORS["robot"]):
    """Draw robot as an arrow."""
    dx = size * np.cos(yaw)
    dy = size * np.sin(yaw)
    ax.annotate("", xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    ax.plot(pos[0], pos[1], "o", color=color, markersize=8, zorder=5)


def plot_single_episode(history, title="Episode", save_name="episode"):
    """Full dashboard for one episode."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"{title} — {history['result']} ({history['steps']} steps)",
                 fontsize=16, fontweight="bold", y=0.98)

    # --- 1. Trajectory (large, top-left) ---
    ax1 = fig.add_axes([0.05, 0.40, 0.42, 0.52])
    draw_arena(ax1)

    positions = history["positions"]
    rewards = history["rewards"]

    # Color trajectory by cumulative reward
    cum_rewards = np.cumsum(rewards)
    norm_rewards = (cum_rewards - cum_rewards.min()) / (cum_rewards.max() - cum_rewards.min() + 1e-8)

    # Draw trajectory as colored segments
    points = positions.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = plt.cm.RdYlGn
    colors = cmap(norm_rewards)
    lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8, zorder=3)
    ax1.add_collection(lc)

    # Start & Goal
    ax1.plot(positions[0, 0], positions[0, 1], "s", color=COLORS["start"],
             markersize=12, label="Start", zorder=6)
    goal = history["goal_pos"]
    goal_circle = plt.Circle(goal, 0.3, color=COLORS["goal"], alpha=0.3, zorder=2)
    ax1.add_patch(goal_circle)
    ax1.plot(goal[0], goal[1], "*", color=COLORS["goal"],
             markersize=15, label="Goal", zorder=6)

    # Robot arrow at final position
    draw_robot(ax1, positions[-1], history["yaws"][-1])

    # Show a few intermediate arrows
    n = len(positions)
    for i in range(0, n, max(1, n // 8)):
        draw_robot(ax1, positions[i], history["yaws"][i], size=0.15,
                   color=(*matplotlib.colors.to_rgb(COLORS["trajectory"]), 0.3))

    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_title("Trajectory (color = cumulative reward)", fontsize=12)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    # --- 2. Reward over time (top-right) ---
    ax2 = fig.add_axes([0.55, 0.72, 0.40, 0.20])
    steps = np.arange(len(rewards))
    ax2.fill_between(steps, rewards, 0, where=rewards >= 0,
                     color=COLORS["reward_pos"], alpha=0.4, label="Positive")
    ax2.fill_between(steps, rewards, 0, where=rewards < 0,
                     color=COLORS["reward_neg"], alpha=0.4, label="Negative")
    ax2.plot(steps, rewards, color="#333", linewidth=0.5, alpha=0.5)
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_title("Step Reward", fontsize=11)
    ax2.set_ylabel("Reward")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2)

    # Cumulative reward
    ax2b = fig.add_axes([0.55, 0.48, 0.40, 0.20])
    ax2b.plot(steps, cum_rewards, color=COLORS["robot"], linewidth=2)
    ax2b.fill_between(steps, cum_rewards, alpha=0.1, color=COLORS["robot"])
    ax2b.set_title("Cumulative Reward", fontsize=11)
    ax2b.set_ylabel("Σ Reward")
    ax2b.set_xlabel("Step")
    ax2b.grid(True, alpha=0.2)

    # --- 3. Sensor readings (bottom-left) ---
    ax3 = fig.add_axes([0.05, 0.05, 0.28, 0.28])
    t = np.arange(len(history["lidar"]))
    ax3.plot(t, history["lidar"], color=COLORS["lidar"], linewidth=1.5, label="LiDAR")
    ax3.axhline(y=10.0, color="gray", linewidth=0.5, linestyle="--", label="Max range")
    ax3.set_title("LiDAR Distance", fontsize=11)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Distance (m)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(-0.5, 11)

    # --- 4. Velocity (bottom-center) ---
    ax4 = fig.add_axes([0.38, 0.05, 0.28, 0.28])
    ax4.plot(t, history["linear_vels"], color=COLORS["robot"], linewidth=1.5, label="Linear vel")
    ax4.plot(t, history["angular_vels"], color=COLORS["start"], linewidth=1.5, alpha=0.7, label="Angular vel")
    ax4.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax4.set_title("Velocities", fontsize=11)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Velocity")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    # --- 5. Goal angle (bottom-right) ---
    ax5 = fig.add_axes([0.71, 0.05, 0.25, 0.28])
    ax5.plot(t, np.degrees(history["goal_angles"]), color=COLORS["goal"], linewidth=1.5)
    ax5.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax5.set_title("Goal Angle", fontsize=11)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Angle (°)")
    ax5.grid(True, alpha=0.2)

    save_path = OUTPUT_DIR / f"{save_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def plot_multi_episodes(histories, title="Multi-Episode", save_name="multi"):
    """Plot multiple episode trajectories on one map."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f"{title} ({len(histories)} episodes)", fontsize=16, fontweight="bold")

    # --- 1. All trajectories ---
    ax = axes[0]
    draw_arena(ax)
    cmap = plt.cm.tab10
    for i, h in enumerate(histories):
        color = cmap(i % 10)
        positions = h["positions"]
        ax.plot(positions[:, 0], positions[:, 1], color=color, linewidth=1.5,
                alpha=0.7, label=f"Ep{i+1} ({h['result'][:3]})")
        ax.plot(positions[0, 0], positions[0, 1], "s", color=color, markersize=6)
        ax.plot(h["goal_pos"][0], h["goal_pos"][1], "*", color=color, markersize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_title("Trajectories", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # --- 2. Reward curves ---
    ax = axes[1]
    for i, h in enumerate(histories):
        cum_r = np.cumsum(h["rewards"])
        ax.plot(cum_r, color=cmap(i % 10), linewidth=1.5, alpha=0.7,
                label=f"Ep{i+1}: {cum_r[-1]:+.1f}")
    ax.set_title("Cumulative Reward", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Σ Reward")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.2)

    # --- 3. Goal distance over time ---
    ax = axes[2]
    for i, h in enumerate(histories):
        dists = np.linalg.norm(h["positions"] - h["goal_pos"], axis=1)
        ax.plot(dists, color=cmap(i % 10), linewidth=1.5, alpha=0.7)
    ax.axhline(y=0.3, color=COLORS["goal"], linewidth=1.5, linestyle="--", label="Goal threshold")
    ax.set_title("Goal Distance", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{save_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Visualize NavigationEnv")
    parser.add_argument("--mode", type=str, default="navigate",
                        choices=["navigate", "random", "forward", "spin", "multi"],
                        help="Visualization mode")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 50)
    print("  Fuzzy-Guided RL AGV — Visualization")
    print("=" * 50)

    env = NavigationEnv()

    if args.mode == "multi":
        print(f"\n  Running {args.episodes} episodes...")
        histories = []
        for i in range(args.episodes):
            h = run_episode(env, controller="navigate", seed=args.seed + i)
            print(f"    Ep{i+1}: {h['result']:9s} | {h['steps']} steps")
            histories.append(h)
        path = plot_multi_episodes(histories, save_name="multi_navigate")
    else:
        print(f"\n  Running {args.mode} episode (seed={args.seed})...")
        h = run_episode(env, controller=args.mode, seed=args.seed)
        print(f"    Result: {h['result']} | {h['steps']} steps | "
              f"Total reward: {sum(h['rewards']):+.2f}")
        path = plot_single_episode(h, title=f"{args.mode.capitalize()} Test",
                                   save_name=f"episode_{args.mode}")

    env.close()
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("=" * 50)
    return path


if __name__ == "__main__":
    main()
