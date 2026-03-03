"""
Fuzzy-Guided Actor-Critic Network.
====================================

Phase 1: 簡潔版 — concat fusion, 確保 state 資訊不被壓掉。
Phase 2: 加回 attention mechanism。

Network Flow:
    state (9) ─┐
               ├─ concat (13) ──→ shared MLP (128, 128) ──→ PolicyHead ──→ action (2)
    fuzzy (4) ─┘                                          └──→ ValueHead  ──→ value (1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from typing import Tuple, Dict, Optional


class FuzzyActorCritic(nn.Module):
    """
    Fuzzy-Guided Actor-Critic (Phase 1 — Concat Fusion).
    
    Usage:
        net = FuzzyActorCritic()
        action, log_prob, value, info = net.get_action(state, fuzzy)
        log_prob, entropy, value = net.evaluate_action(state, fuzzy, action)
    """

    def __init__(
        self,
        state_dim: int = 9,
        fuzzy_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        **kwargs,  # ignore unused args from config
    ):
        super().__init__()
        self.state_dim = state_dim
        self.fuzzy_dim = fuzzy_dim
        self.action_dim = action_dim

        input_dim = state_dim + fuzzy_dim  # 13

        # ---- Shared feature extractor ----
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ---- Policy head (Actor) ----
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),   # action mean in [-1, 1]
        )

        # Learnable log_std with clamp
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.LOG_STD_MIN = -2.0
        self.LOG_STD_MAX = 0.5

        # ---- Value head (Critic) ----
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_features(self, state: torch.Tensor, fuzzy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, fuzzy], dim=-1)
        return self.feature_net(x)

    def forward(self, state: torch.Tensor, fuzzy: torch.Tensor):
        features = self._get_features(state, fuzzy)
        action_mean = self.policy_net(features)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        action_std = log_std.exp().expand_as(action_mean)
        value = self.value_net(features)
        return action_mean, action_std, value

    def get_action(
        self,
        state: torch.Tensor,
        fuzzy: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            fuzzy = fuzzy.unsqueeze(0)
            squeeze = True

        action_mean, action_std, value = self.forward(state, fuzzy)

        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()

        action_clipped = action.clamp(-1.0, 1.0)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = value.squeeze(-1)

        if squeeze:
            action_clipped = action_clipped.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        info = {
            "attn_weights": torch.tensor([[0.5, 0.5]]),  # placeholder
            "action_mean": action_mean,
            "action_std": action_std,
        }
        return action_clipped, log_prob, value, info

    def evaluate_action(
        self,
        state: torch.Tensor,
        fuzzy: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, value = self.forward(state, fuzzy)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = value.squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, state: torch.Tensor, fuzzy: torch.Tensor) -> torch.Tensor:
        features = self._get_features(state, fuzzy)
        return self.value_net(features).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (f"FuzzyActorCritic(state={self.state_dim}, fuzzy={self.fuzzy_dim}, "
                f"action={self.action_dim}, params={self.count_parameters():,})")