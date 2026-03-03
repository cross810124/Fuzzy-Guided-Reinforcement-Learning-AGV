"""
PPO Agent with Fuzzy Guidance Integration.
==========================================

完整的 Proximal Policy Optimization 實作，整合：
1. Fuzzy-Guided Actor-Critic Network (multi-modal input)
2. Fuzzy Enhanced Reward (reward shaping)
3. Safety-Constrained Exploration (fuzzy → action filtering)
4. GAE (Generalized Advantage Estimation)

Usage:
    agent = PPOAgent(config)
    
    # Collect rollout
    for step in range(n_steps):
        action = agent.select_action(obs, fuzzy_guidance)
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, done, fuzzy_guidance, ...)
        obs = next_obs
    
    # Update
    metrics = agent.update()
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from agent.actor_critic import FuzzyActorCritic
from agent.fuzzy_reward import FuzzyRewardShaper
from fuzzy.inference_system import FuzzyInferenceSystem, FuzzyGuidance


# ================================================================
# Config
# ================================================================

@dataclass
class PPOConfig:
    """PPO 超參數設定。"""
    # Environment
    state_dim: int = 9
    fuzzy_dim: int = 4
    action_dim: int = 2

    # Network
    state_hidden: int = 64
    fuzzy_hidden: int = 32
    fused_dim: int = 64

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Rollout
    n_steps: int = 2048         # steps per rollout
    n_epochs: int = 10          # PPO update epochs
    batch_size: int = 64        # mini-batch size

    # Fuzzy reward
    lambda_safety: float = 0.05
    lambda_exploration: float = 0.02
    lambda_confidence: float = 0.03

    # Safety constraints
    safety_threshold: float = 0.3   # safety_level < this → limit action
    speed_limit_factor: float = 0.5 # scale action when unsafe

    # Device
    device: str = "auto"


# ================================================================
# Rollout Buffer
# ================================================================

class RolloutBuffer:
    """存儲 rollout 軌跡。"""

    def __init__(self, n_steps: int, state_dim: int, fuzzy_dim: int,
                 action_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.pos = 0
        self.full = False

        self.states = np.zeros((n_steps, state_dim), dtype=np.float32)
        self.fuzzy_signals = np.zeros((n_steps, fuzzy_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

    def add(self, state, fuzzy, action, log_prob, reward, value, done):
        self.states[self.pos] = state
        self.fuzzy_signals[self.pos] = fuzzy
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE advantages and discounted returns."""
        last_gae = 0.0
        n = self.pos if not self.full else self.n_steps

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (self.rewards[t] + gamma * next_value * next_non_terminal
                     - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """生成 mini-batch iterator。"""
        n = self.pos if not self.full else self.n_steps
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n:
                break
            batch_idx = indices[start:end]

            yield {
                "states": torch.FloatTensor(self.states[batch_idx]).to(self.device),
                "fuzzy": torch.FloatTensor(self.fuzzy_signals[batch_idx]).to(self.device),
                "actions": torch.FloatTensor(self.actions[batch_idx]).to(self.device),
                "old_log_probs": torch.FloatTensor(self.log_probs[batch_idx]).to(self.device),
                "advantages": torch.FloatTensor(self.advantages[batch_idx]).to(self.device),
                "returns": torch.FloatTensor(self.returns[batch_idx]).to(self.device),
            }

    def reset(self):
        self.pos = 0
        self.full = False


# ================================================================
# PPO Agent
# ================================================================

class PPOAgent:
    """
    PPO Agent with Fuzzy Guidance.
    
    整合：
    - FuzzyActorCritic: multi-modal 網路
    - FuzzyInferenceSystem: 模糊推論
    - FuzzyRewardShaper: 獎勵塑形
    - Safety filtering: 動作安全限制
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.device = self._get_device()

        # Network
        self.network = FuzzyActorCritic(
            state_dim=self.config.state_dim,
            fuzzy_dim=self.config.fuzzy_dim,
            action_dim=self.config.action_dim,
            state_hidden=self.config.state_hidden,
            fuzzy_hidden=self.config.fuzzy_hidden,
            fused_dim=self.config.fused_dim,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.lr,
            eps=1e-5,
        )

        # Fuzzy systems
        self.fis = FuzzyInferenceSystem()
        self.reward_shaper = FuzzyRewardShaper(
            lambda_safety=self.config.lambda_safety,
            lambda_exploration=self.config.lambda_exploration,
            lambda_confidence=self.config.lambda_confidence,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=self.config.n_steps,
            state_dim=self.config.state_dim,
            fuzzy_dim=self.config.fuzzy_dim,
            action_dim=self.config.action_dim,
            device=self.device,
        )

        # Training stats
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, FuzzyGuidance, Dict]:
        """
        選擇動作（完整 pipeline）。
        
        Flow: obs → FIS → fuzzy_guidance
              obs + fuzzy → Network → raw_action
              raw_action + safety_filter → final_action
        
        Args:
            obs: observation (9-dim)
            deterministic: eval mode
            
        Returns:
            action: np.ndarray (2-dim), clipped to [-1, 1]
            guidance: FuzzyGuidance
            info: {log_prob, value, attn_weights, ...}
        """
        # Step 1: Fuzzy inference
        guidance = self.fis.infer_from_obs(obs)
        fuzzy_signal = guidance.to_array()

        # Step 2: Network forward
        state_t = torch.FloatTensor(obs).to(self.device)
        fuzzy_t = torch.FloatTensor(fuzzy_signal).to(self.device)

        action_t, log_prob, value, net_info = self.network.get_action(
            state_t, fuzzy_t, deterministic=deterministic
        )

        action = action_t.cpu().numpy()

        # Step 3: Safety-constrained action filtering
        action = self._safety_filter(action, guidance)

        info = {
            "log_prob": log_prob.cpu().item(),
            "value": value.cpu().item(),
            "guidance": guidance,
            "fuzzy_signal": fuzzy_signal,
            "attn_weights": net_info["attn_weights"].cpu().numpy(),
        }

        return action, guidance, info

    def _safety_filter(self, action: np.ndarray, guidance: FuzzyGuidance) -> np.ndarray:
        """
        Phase 1: 不做任何 filtering，讓 agent 自由學習。
        Phase 2: 啟用 safety constraints。
        """
        return np.clip(action, -1.0, 1.0)

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        base_reward: float,
        done: bool,
        guidance: FuzzyGuidance,
        action_info: Dict,
    ):
        """
        存儲一個 transition 到 buffer。
        
        會自動計算 fuzzy enhanced reward。
        """
        # Compute fuzzy-enhanced reward
        # Phase 1: 直接用 base reward，fuzzy reward shaping 先關閉
        # 確保 agent 先學會基本導航，Phase 2 再加入 fuzzy reward
        total_reward = base_reward

        self.buffer.add(
            state=obs,
            fuzzy=guidance.to_array(),
            action=action,
            log_prob=action_info["log_prob"],
            reward=total_reward,
            value=action_info["value"],
            done=float(done),
        )

        self.total_steps += 1

    def end_episode(self, success: bool, episode_reward: float = 0.0):
        """Episode 結束時呼叫。"""
        self.reward_shaper.end_episode(success, episode_reward)
        self.total_episodes += 1

    @torch.no_grad()
    def _compute_last_value(self, last_obs: np.ndarray) -> float:
        """計算最後一個 state 的 value（用於 GAE bootstrap）。"""
        guidance = self.fis.infer_from_obs(last_obs)
        state_t = torch.FloatTensor(last_obs).to(self.device)
        fuzzy_t = torch.FloatTensor(guidance.to_array()).to(self.device)
        value = self.network.get_value(state_t.unsqueeze(0), fuzzy_t.unsqueeze(0))
        return value.cpu().item()

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """
        執行 PPO 更新。
        
        Args:
            last_obs: 最後一個 observation（用於 GAE bootstrap）
            
        Returns:
            metrics: {policy_loss, value_loss, entropy, clip_fraction, ...}
        """
        # Compute GAE
        last_value = self._compute_last_value(last_obs)
        self.buffer.compute_gae(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Normalize advantages
        n = self.buffer.pos if not self.buffer.full else self.buffer.n_steps
        adv = self.buffer.advantages[:n]
        adv_mean, adv_std = adv.mean(), adv.std()
        if adv_std > 1e-8:
            self.buffer.advantages[:n] = (adv - adv_mean) / (adv_std + 1e-8)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        num_batches = 0

        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                # Evaluate actions
                new_log_prob, entropy, new_value = self.network.evaluate_action(
                    batch["states"], batch["fuzzy"], batch["actions"]
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_log_prob - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps,
                                    1 + self.config.clip_eps) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_value, batch["returns"])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss
                        + self.config.value_coef * value_loss
                        + self.config.entropy_coef * entropy_loss)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(),
                                         self.config.max_grad_norm)
                self.optimizer.step()

                # Track stats
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_frac += clip_frac.item()
                num_batches += 1

        # Reset buffer
        self.buffer.reset()
        self.update_count += 1

        metrics = {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
            "clip_fraction": total_clip_frac / max(num_batches, 1),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "update_count": self.update_count,
            "success_rate": self.reward_shaper.success_rate,
        }

        return metrics

    def save(self, path: str):
        """儲存 checkpoint。"""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "update_count": self.update_count,
        }, path)

    def load(self, path: str):
        """載入 checkpoint。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.total_episodes = ckpt.get("total_episodes", 0)
        self.update_count = ckpt.get("update_count", 0)

    def summary(self) -> str:
        lines = [
            "PPOAgent(",
            f"  device={self.device}",
            f"  network={self.network.summary()}",
            f"  fuzzy_rules={len(self.fis.rule_base)}",
            f"  n_steps={self.config.n_steps}, batch_size={self.config.batch_size}",
            f"  lr={self.config.lr}, gamma={self.config.gamma}",
            ")",
        ]
        return "\n".join(lines)


# Need F for mse_loss
import torch.nn.functional as F