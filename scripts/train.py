#!/usr/bin/env python3
"""
Fuzzy-Guided RL AGV — Training Script.
========================================

完整訓練循環：
    1. Rollout collection (PPO on-policy)
    2. Fuzzy inference + reward shaping (每步)
    3. PPO update
    4. 定期 evaluation + checkpoint
    5. TensorBoard logging

Usage:
    python scripts/train.py                                # 預設 config
    python scripts/train.py --config config/default.yaml   # 指定 config
    python scripts/train.py --total_timesteps 200000       # CLI 覆蓋
    python scripts/train.py --seed 123                     # 指定 seed
    python scripts/train.py --resume data/checkpoints/latest.pt  # 續訓

TensorBoard:
    tensorboard --logdir data/logs
"""

import sys
import os
import time
import argparse
import datetime
from pathlib import Path
from collections import deque

import yaml
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.mujoco.navigation_env import NavigationEnv
from agent.ppo_agent import PPOAgent, PPOConfig
from fuzzy.inference_system import FuzzyGuidance


# ================================================================
# Config Loading
# ================================================================

def load_config(config_path: str = None) -> dict:
    """載入 YAML config。"""
    default_path = PROJECT_ROOT / "config" / "default.yaml"
    path = Path(config_path) if config_path else default_path

    if not path.exists():
        print(f"  [WARN] Config not found: {path}, using defaults")
        return {}

    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def build_ppo_config(cfg: dict) -> PPOConfig:
    """從 dict 建立 PPOConfig。"""
    ppo_cfg = cfg.get("ppo", {})
    net_cfg = cfg.get("network", {})
    fuzzy_cfg = cfg.get("fuzzy_reward", {})
    safety_cfg = cfg.get("safety", {})

    return PPOConfig(
        lr=ppo_cfg.get("lr", 3e-4),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_eps=ppo_cfg.get("clip_eps", 0.2),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
        value_coef=ppo_cfg.get("value_coef", 0.5),
        n_steps=ppo_cfg.get("n_steps", 2048),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        batch_size=ppo_cfg.get("batch_size", 64),
        state_hidden=net_cfg.get("state_hidden", 64),
        fuzzy_hidden=net_cfg.get("fuzzy_hidden", 32),
        fused_dim=net_cfg.get("fused_dim", 64),
        lambda_safety=fuzzy_cfg.get("lambda_safety", 0.05),
        lambda_exploration=fuzzy_cfg.get("lambda_exploration", 0.02),
        lambda_confidence=fuzzy_cfg.get("lambda_confidence", 0.03),
        safety_threshold=safety_cfg.get("safety_threshold", 0.3),
        speed_limit_factor=safety_cfg.get("speed_limit_factor", 0.5),
    )


# ================================================================
# TensorBoard Logger
# ================================================================

class Logger:
    """訓練 Logger（TensorBoard + Console）。"""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir))
        except ImportError:
            print("  [WARN] TensorBoard not available, logging to console only")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict, step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, values, step)

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()


# ================================================================
# Evaluation
# ================================================================

def evaluate(agent: PPOAgent, env: NavigationEnv, n_episodes: int = 10,
             seed: int = 10000) -> dict:
    """
    評估 agent 效能。
    
    Returns:
        metrics: {success_rate, collision_rate, avg_reward, avg_steps,
                  avg_goal_dist, path_efficiency}
    """
    successes = 0
    collisions = 0
    total_rewards = []
    total_steps = []
    path_efficiencies = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        goal_pos = info["goal_pos"]
        start_pos = obs[:2].copy()
        shortest_dist = np.linalg.norm(start_pos - goal_pos)

        ep_reward = 0.0
        ep_steps = 0
        path_length = 0.0
        prev_pos = start_pos.copy()

        done = False
        while not done:
            action, guidance, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_steps += 1
            path_length += np.linalg.norm(obs[:2] - prev_pos)
            prev_pos = obs[:2].copy()

        if info.get("success"):
            successes += 1
            # Path efficiency = actual path / shortest path
            if shortest_dist > 0.1:
                path_efficiencies.append(path_length / shortest_dist)
        if info.get("collision"):
            collisions += 1

        total_rewards.append(ep_reward)
        total_steps.append(ep_steps)

    return {
        "success_rate": successes / n_episodes,
        "collision_rate": collisions / n_episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_steps": np.mean(total_steps),
        "avg_goal_dist": np.mean([
            np.linalg.norm(obs[:2] - goal_pos)  # last episode only, for quick check
        ]),
        "path_efficiency": np.mean(path_efficiencies) if path_efficiencies else 0.0,
    }


# ================================================================
# Training Loop
# ================================================================

def train(args):
    """主訓練函數。"""
    # ---- Load config ----
    cfg = load_config(args.config)

    # CLI overrides
    total_timesteps = args.total_timesteps or cfg.get("total_timesteps", 500000)
    seed = args.seed or cfg.get("seed", 42)
    eval_interval = cfg.get("eval_interval", 10)
    eval_episodes = cfg.get("eval_episodes", 10)
    save_interval = cfg.get("save_interval", 50)
    log_interval = cfg.get("log_interval", 1)

    output_cfg = cfg.get("output", {})
    log_dir = output_cfg.get("log_dir", "data/logs")
    checkpoint_dir = output_cfg.get("checkpoint_dir", "data/checkpoints")
    experiment_name = output_cfg.get("experiment_name", "fuzzy_rl_mvp")

    # ---- Setup ----
    ppo_config = build_ppo_config(cfg)
    agent = PPOAgent(ppo_config)

    env = NavigationEnv()
    eval_env = NavigationEnv()

    # Resolve paths relative to project root
    log_dir = str(PROJECT_ROOT / log_dir)
    checkpoint_dir = str(PROJECT_ROOT / checkpoint_dir)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logger = Logger(log_dir, experiment_name)

    # Resume from checkpoint
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        agent.load(args.resume)

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Print info ----
    print("=" * 60)
    print("  Fuzzy-Guided RL AGV — Training")
    print("=" * 60)
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"  Steps per update: {ppo_config.n_steps}")
    print(f"  Total updates:    ~{total_timesteps // ppo_config.n_steps}")
    print(f"  Device:           {agent.device}")
    print(f"  Parameters:       {agent.network.count_parameters():,}")
    print(f"  Seed:             {seed}")
    print(f"  Log dir:          {logger.run_dir}")
    print(f"  Checkpoint dir:   {checkpoint_dir}")
    print("=" * 60)

    # ---- Training loop ----
    obs, info = env.reset(seed=seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    best_success_rate = 0.0

    # Rolling stats
    recent_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    recent_collisions = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)

    train_start = time.time()
    update_count = 0

    while agent.total_steps < total_timesteps:
        # ---- Collect rollout ----
        for step in range(ppo_config.n_steps):
            action, guidance, action_info = agent.select_action(obs)
            next_obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, done, guidance, action_info)
            episode_reward += reward
            episode_steps += 1

            if done:
                success = env_info.get("success", False)
                collision = env_info.get("collision", False)

                agent.end_episode(success, episode_reward)
                episode_count += 1

                recent_rewards.append(episode_reward)
                recent_successes.append(1.0 if success else 0.0)
                recent_collisions.append(1.0 if collision else 0.0)
                recent_lengths.append(episode_steps)

                # Log episode
                logger.log_scalar("episode/reward", episode_reward, episode_count)
                logger.log_scalar("episode/length", episode_steps, episode_count)
                logger.log_scalar("episode/success", float(success), episode_count)
                logger.log_scalar("episode/collision", float(collision), episode_count)

                episode_reward = 0.0
                episode_steps = 0
                obs, info = env.reset()
            else:
                obs = next_obs

        # ---- PPO Update ----
        metrics = agent.update(last_obs=obs)
        update_count += 1

        # Log training metrics
        logger.log_scalar("train/policy_loss", metrics["policy_loss"], update_count)
        logger.log_scalar("train/value_loss", metrics["value_loss"], update_count)
        logger.log_scalar("train/entropy", metrics["entropy"], update_count)
        logger.log_scalar("train/clip_fraction", metrics["clip_fraction"], update_count)
        logger.log_scalar("train/total_steps", agent.total_steps, update_count)

        if recent_rewards:
            logger.log_scalar("rollout/mean_reward", np.mean(recent_rewards), update_count)
            logger.log_scalar("rollout/success_rate", np.mean(recent_successes), update_count)
            logger.log_scalar("rollout/collision_rate", np.mean(recent_collisions), update_count)
            logger.log_scalar("rollout/mean_length", np.mean(recent_lengths), update_count)

        # ---- Console log ----
        if update_count % log_interval == 0:
            elapsed = time.time() - train_start
            fps = agent.total_steps / elapsed if elapsed > 0 else 0

            mean_r = np.mean(recent_rewards) if recent_rewards else 0
            success_r = np.mean(recent_successes) if recent_successes else 0
            collision_r = np.mean(recent_collisions) if recent_collisions else 0

            print(
                f"  Update {update_count:4d} | "
                f"Steps {agent.total_steps:7,d}/{total_timesteps:,d} | "
                f"Ep {episode_count:4d} | "
                f"R={mean_r:+7.1f} | "
                f"Succ={success_r:.0%} | "
                f"Coll={collision_r:.0%} | "
                f"PL={metrics['policy_loss']:.4f} | "
                f"VL={metrics['value_loss']:.4f} | "
                f"Ent={metrics['entropy']:.3f} | "
                f"FPS={fps:.0f}"
            )

        # ---- Evaluation ----
        if update_count % eval_interval == 0:
            eval_metrics = evaluate(agent, eval_env, n_episodes=eval_episodes)

            print(f"\n  {'─' * 50}")
            print(f"  EVAL (update {update_count}):")
            print(f"    Success rate:   {eval_metrics['success_rate']:.0%}")
            print(f"    Collision rate: {eval_metrics['collision_rate']:.0%}")
            print(f"    Avg reward:     {eval_metrics['avg_reward']:+.1f}")
            print(f"    Avg steps:      {eval_metrics['avg_steps']:.0f}")
            print(f"    Path efficiency:{eval_metrics['path_efficiency']:.2f}")
            print(f"  {'─' * 50}\n")

            for key, val in eval_metrics.items():
                logger.log_scalar(f"eval/{key}", val, update_count)

            # Save best model
            if eval_metrics["success_rate"] > best_success_rate:
                best_success_rate = eval_metrics["success_rate"]
                best_path = Path(checkpoint_dir) / "best.pt"
                agent.save(str(best_path))
                print(f"  ★ New best model! Success rate: {best_success_rate:.0%}")

        # ---- Checkpoint ----
        if update_count % save_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_{update_count:05d}.pt"
            agent.save(str(ckpt_path))
            # Also save as latest
            latest_path = Path(checkpoint_dir) / "latest.pt"
            agent.save(str(latest_path))

        logger.flush()

    # ---- Training complete ----
    total_time = time.time() - train_start

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total timesteps: {agent.total_steps:,}")
    print(f"  Total episodes:  {episode_count}")
    print(f"  Total updates:   {update_count}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"  Best success:    {best_success_rate:.0%}")
    print(f"  Final mean R:    {np.mean(recent_rewards) if recent_rewards else 0:+.1f}")

    # Final save
    final_path = Path(checkpoint_dir) / "final.pt"
    agent.save(str(final_path))
    print(f"\n  Saved final model: {final_path}")

    # Final eval
    print(f"\n  Running final evaluation ({eval_episodes} episodes)...")
    final_eval = evaluate(agent, eval_env, n_episodes=eval_episodes)
    print(f"    Success rate:   {final_eval['success_rate']:.0%}")
    print(f"    Collision rate: {final_eval['collision_rate']:.0%}")
    print(f"    Avg reward:     {final_eval['avg_reward']:+.1f}")

    # Save training summary
    summary = {
        "total_timesteps": agent.total_steps,
        "total_episodes": episode_count,
        "total_updates": update_count,
        "total_time_sec": total_time,
        "best_success_rate": best_success_rate,
        "final_eval": final_eval,
    }
    results_dir = PROJECT_ROOT / output_cfg.get("results_dir", "data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "training_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"  Saved summary: {summary_path}")

    logger.close()
    env.close()
    eval_env.close()

    print("=" * 60)


# ================================================================
# Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Fuzzy-Guided RL AGV")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override total timesteps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
