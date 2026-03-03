#!/usr/bin/env python3
"""
Test Fuzzy-Guided Actor-Critic & PPO Agent.
=============================================

Usage:
    python scripts/test_agent.py                 # 全部測試
    python scripts/test_agent.py --mode network  # 只測網路
    python scripts/test_agent.py --mode reward   # 只測 reward shaper
    python scripts/test_agent.py --mode agent    # 只測 PPO agent
    python scripts/test_agent.py --mode rollout  # 跑一個完整 rollout + update
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.actor_critic import FuzzyActorCritic
from agent.fuzzy_reward import FuzzyRewardShaper
from agent.ppo_agent import PPOAgent, PPOConfig
from fuzzy.inference_system import FuzzyInferenceSystem, FuzzyGuidance


def test_network():
    """測試 FuzzyActorCritic 網路。"""
    print("\n  [Test] FuzzyActorCritic Network")

    net = FuzzyActorCritic()
    print(f"    {net.summary()}")

    # Test forward pass
    batch_size = 8
    state = torch.randn(batch_size, 9)
    fuzzy = torch.rand(batch_size, 4)

    action_mean, action_std, value, attn = net(state, fuzzy)
    print(f"    Forward pass:")
    print(f"      action_mean: {action_mean.shape} range=[{action_mean.min():.3f}, {action_mean.max():.3f}]")
    print(f"      action_std:  {action_std.shape} range=[{action_std.min():.3f}, {action_std.max():.3f}]")
    print(f"      value:       {value.shape}")
    print(f"      attn:        {attn.shape} → state={attn[0,0]:.3f}, fuzzy={attn[0,1]:.3f}")

    assert action_mean.shape == (batch_size, 2)
    assert action_std.shape == (batch_size, 2)
    assert value.shape == (batch_size, 1)
    assert attn.shape == (batch_size, 2)
    assert (action_mean >= -1).all() and (action_mean <= 1).all(), "action_mean out of [-1,1]"
    assert (action_std > 0).all(), "action_std must be positive"
    assert torch.allclose(attn.sum(dim=-1), torch.ones(batch_size)), "attn weights must sum to 1"

    # Test get_action (single obs)
    state_single = torch.randn(9)
    fuzzy_single = torch.rand(4)
    action, log_prob, value, info = net.get_action(state_single, fuzzy_single)
    print(f"\n    get_action (single):")
    print(f"      action: {action.shape} = [{action[0]:.3f}, {action[1]:.3f}]")
    print(f"      log_prob: {log_prob.shape} = {log_prob:.3f}")
    print(f"      value: {value.shape} = {value:.3f}")
    assert action.shape == (2,)
    assert log_prob.shape == ()
    assert value.shape == ()

    # Test get_action (deterministic)
    action_det, _, _, _ = net.get_action(state_single, fuzzy_single, deterministic=True)
    action_det2, _, _, _ = net.get_action(state_single, fuzzy_single, deterministic=True)
    assert torch.allclose(action_det, action_det2), "Deterministic should be consistent"
    print(f"    deterministic: consistent ✓")

    # Test evaluate_action
    state_batch = torch.randn(batch_size, 9)
    fuzzy_batch = torch.rand(batch_size, 4)
    action_batch = torch.randn(batch_size, 2).clamp(-1, 1)
    log_prob, entropy, value = net.evaluate_action(state_batch, fuzzy_batch, action_batch)
    print(f"\n    evaluate_action:")
    print(f"      log_prob: {log_prob.shape}")
    print(f"      entropy:  {entropy.shape}, mean={entropy.mean():.3f}")
    print(f"      value:    {value.shape}")
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size,)

    # Test gradient flow
    loss = log_prob.mean() + value.mean()
    loss.backward()
    grad_ok = all(p.grad is not None for p in net.parameters() if p.requires_grad)
    print(f"\n    Gradient flow: {'✓' if grad_ok else '✗'}")
    assert grad_ok

    print("    ✓ All network tests passed")


def test_reward_shaper():
    """測試 FuzzyRewardShaper。"""
    print("\n  [Test] FuzzyRewardShaper")

    shaper = FuzzyRewardShaper()

    # Create fake guidance
    guidance = FuzzyGuidance(
        safety_level=0.3,
        exploration_encouragement=0.7,
        action_confidence=0.6,
        speed_modifier=0.5,
    )

    obs = np.array([0, 0, 0, 0.5, 0, 3, 4, 1.5, 0.2], dtype=np.float32)

    # Test compute
    result = shaper.compute(base_reward=-0.5, guidance=guidance, obs=obs)
    print(f"    Reward breakdown:")
    for key, val in result.items():
        print(f"      {key}: {val:+.4f}")

    assert "r_safety" in result
    assert "r_exploration" in result
    assert "r_confidence" in result
    assert "r_total" in result
    assert result["r_safety"] >= 0, "Safety reward when near obstacle should be positive"
    print(f"    r_safety > 0 (near obstacle): ✓")

    # Test with far obstacle
    obs_far = obs.copy()
    obs_far[7] = 8.0  # far obstacle
    result_far = shaper.compute(base_reward=-0.5, guidance=guidance, obs=obs_far)
    assert result_far["r_safety"] == 0.0, "No safety reward when far from obstacle"
    print(f"    r_safety = 0 (far obstacle): ✓")

    # Test novelty
    shaper._visited_cells.clear()  # Reset novelty tracking
    result1 = shaper.compute(base_reward=0, guidance=guidance, obs=obs)
    result2 = shaper.compute(base_reward=0, guidance=guidance, obs=obs)
    assert result1["novelty"] == 1.0, "First visit should be novel"
    assert result2["novelty"] == 0.0, "Second visit should NOT be novel"
    print(f"    Novelty tracking: ✓")

    # Test episode tracking
    shaper.end_episode(success=True, episode_reward=10.0)
    shaper.end_episode(success=False, episode_reward=-5.0)
    shaper.end_episode(success=True, episode_reward=8.0)
    stats = shaper.get_stats()
    print(f"    Success rate: {stats['success_rate']:.2f} (expected ~0.67)")
    assert 0.6 < stats["success_rate"] < 0.7

    print("    ✓ All reward shaper tests passed")


def test_ppo_agent():
    """測試 PPO Agent 初始化和基本功能。"""
    print("\n  [Test] PPO Agent")

    config = PPOConfig(n_steps=64, batch_size=16, n_epochs=2)
    agent = PPOAgent(config)
    print(f"    {agent.summary()}")

    # Test select_action
    obs = np.array([0, 0, 0, 0.5, 0, 3, 4, 5.0, 0.2], dtype=np.float32)
    action, guidance, info = agent.select_action(obs)
    print(f"\n    select_action:")
    print(f"      action: [{action[0]:+.3f}, {action[1]:+.3f}]")
    print(f"      guidance: {guidance}")
    print(f"      attn: state={info['attn_weights'][0,0]:.3f}, fuzzy={info['attn_weights'][0,1]:.3f}")

    assert action.shape == (2,)
    assert -1 <= action[0] <= 1 and -1 <= action[1] <= 1
    assert isinstance(guidance, FuzzyGuidance)

    # Test safety filter
    # Create dangerous scenario
    obs_danger = np.array([0, 0, 0, 0.8, 0, 3, 4, 0.3, 0.5], dtype=np.float32)
    action_danger, guidance_danger, _ = agent.select_action(obs_danger)
    print(f"\n    Safety filter (obstacle=0.3m):")
    print(f"      safety_level: {guidance_danger.safety_level:.3f}")
    print(f"      action magnitude: {np.abs(action_danger).mean():.3f}")

    # Test store_transition
    agent.store_transition(obs, action, -0.5, False, guidance, info)
    assert agent.buffer.pos == 1
    print(f"\n    store_transition: buffer.pos={agent.buffer.pos} ✓")

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name
    agent.save(save_path)
    agent2 = PPOAgent(config)
    agent2.load(save_path)
    os.unlink(save_path)
    print(f"    save/load: ✓")

    print("    ✓ All PPO agent tests passed")


def test_rollout_and_update():
    """測試完整的 rollout + PPO update。"""
    print("\n  [Test] Full Rollout + Update")

    from envs.mujoco.navigation_env import NavigationEnv

    config = PPOConfig(
        n_steps=128,
        batch_size=32,
        n_epochs=3,
    )
    agent = PPOAgent(config)
    env = NavigationEnv()

    print(f"    Config: n_steps={config.n_steps}, batch={config.batch_size}, "
          f"epochs={config.n_epochs}")
    print(f"    Device: {agent.device}")

    # Collect rollout
    obs, info = env.reset(seed=42)
    episode_reward = 0.0
    episodes_completed = 0

    for step in range(config.n_steps):
        action, guidance, action_info = agent.select_action(obs)
        next_obs, reward, terminated, truncated, env_info = env.step(action)
        done = terminated or truncated

        agent.store_transition(obs, action, reward, done, guidance, action_info)
        episode_reward += reward

        if done:
            success = env_info.get("success", False)
            agent.end_episode(success, episode_reward)
            episodes_completed += 1
            episode_reward = 0.0
            obs, info = env.reset()
        else:
            obs = next_obs

    print(f"    Rollout: {config.n_steps} steps, {episodes_completed} episodes completed")

    # PPO update
    metrics = agent.update(last_obs=obs)
    print(f"\n    PPO Update metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"      {key}: {val:.4f}")
        else:
            print(f"      {key}: {val}")

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert metrics["update_count"] == 1
    assert agent.buffer.pos == 0, "Buffer should be reset after update"

    env.close()
    print("    ✓ Full rollout + update test passed")


def main():
    parser = argparse.ArgumentParser(description="Test Agent")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "network", "reward", "agent", "rollout"])
    args = parser.parse_args()

    print("=" * 55)
    print("  Fuzzy-Guided Actor-Critic — Test Suite")
    print("=" * 55)

    if args.mode in ("all", "network"):
        test_network()

    if args.mode in ("all", "reward"):
        test_reward_shaper()

    if args.mode in ("all", "agent"):
        test_ppo_agent()

    if args.mode in ("all", "rollout"):
        test_rollout_and_update()

    print("\n" + "=" * 55)
    print("  All tests completed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
