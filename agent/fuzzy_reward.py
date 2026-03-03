"""
Fuzzy Enhanced Reward.
======================

論文公式實作：
    R_total = R_base + R_fuzzy + R_adaptive
    
    R_fuzzy = R_safety + R_exploration + R_confidence
    
    R_safety      = λ1 × safety_level × I_near_obstacle
    R_exploration = λ2 × exploration_encouragement × novelty(s)
    R_confidence  = λ3 × action_confidence × success_rate
    
    R_adaptive    = η × sign(performance_trend) × fuzzy_contribution

係數 (論文):
    λ1 = 0.05 (safety)
    λ2 = 0.02 (exploration)
    λ3 = 0.03 (confidence)
    η  = 0.01 (adaptive)

Phase 1 簡化:
    - novelty(s) = 基於 state visitation count 的簡易版
    - success_rate = 滾動窗口成功率
    - R_adaptive 先關閉（Phase 2 啟用）
"""

from typing import Dict, Optional
from collections import deque

import numpy as np


class FuzzyRewardShaper:
    """
    模糊增強獎勵計算器。
    
    Usage:
        shaper = FuzzyRewardShaper()
        
        # Each step:
        fuzzy_reward = shaper.compute(
            base_reward=reward,
            guidance=guidance,       # FuzzyGuidance object
            obs=obs,
            info=info,
        )
        total_reward = base_reward + fuzzy_reward
        
        # End of episode:
        shaper.end_episode(success=True/False)
    """

    def __init__(
        self,
        lambda_safety: float = 0.05,
        lambda_exploration: float = 0.02,
        lambda_confidence: float = 0.03,
        eta_adaptive: float = 0.01,
        near_obstacle_threshold: float = 2.0,
        novelty_grid_size: float = 0.5,
        success_window: int = 100,
        enable_adaptive: bool = False,  # Phase 2 啟用
    ):
        # Fuzzy reward coefficients (from paper)
        self.lambda_safety = lambda_safety
        self.lambda_exploration = lambda_exploration
        self.lambda_confidence = lambda_confidence
        self.eta_adaptive = eta_adaptive

        # Thresholds
        self.near_obstacle_threshold = near_obstacle_threshold
        self.novelty_grid_size = novelty_grid_size
        self.enable_adaptive = enable_adaptive

        # State tracking
        self._visited_cells = set()       # For novelty
        self._success_history = deque(maxlen=success_window)
        self._episode_rewards = deque(maxlen=success_window)
        self._success_rate = 0.0

    def compute(
        self,
        base_reward: float,
        guidance: "FuzzyGuidance",
        obs: np.ndarray,
        info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        計算模糊增強獎勵。
        
        Args:
            base_reward: 環境原始獎勵
            guidance: FuzzyGuidance (from FIS)
            obs: observation vector (9-dim)
            info: environment info dict
            
        Returns:
            dict with {r_safety, r_exploration, r_confidence, 
                       r_adaptive, r_fuzzy_total, r_total}
        """
        # Extract fuzzy signals
        safety = guidance.safety_level
        exploration = guidance.exploration_encouragement
        confidence = guidance.action_confidence
        speed = guidance.speed_modifier

        # Extract state info
        obstacle_dist = float(obs[7])  # lidar distance
        pos = obs[:2]

        # ---- R_safety ----
        # I_near_obstacle: 1 if obstacle is near, 0 otherwise
        near_obstacle = 1.0 if obstacle_dist < self.near_obstacle_threshold else 0.0
        r_safety = self.lambda_safety * safety * near_obstacle

        # ---- R_exploration ----
        # novelty(s): based on position visitation (grid-based)
        novelty = self._compute_novelty(pos)
        r_exploration = self.lambda_exploration * exploration * novelty

        # ---- R_confidence ----
        r_confidence = self.lambda_confidence * confidence * self._success_rate

        # ---- R_fuzzy total ----
        r_fuzzy = r_safety + r_exploration + r_confidence

        # ---- R_adaptive (Phase 2) ----
        r_adaptive = 0.0
        if self.enable_adaptive:
            perf_trend = self._compute_performance_trend()
            fuzzy_contribution = r_fuzzy
            r_adaptive = self.eta_adaptive * np.sign(perf_trend) * abs(fuzzy_contribution)

        # ---- R_total ----
        r_total = base_reward + r_fuzzy + r_adaptive

        return {
            "r_safety": r_safety,
            "r_exploration": r_exploration,
            "r_confidence": r_confidence,
            "r_fuzzy_total": r_fuzzy,
            "r_adaptive": r_adaptive,
            "r_total": r_total,
            "base_reward": base_reward,
            "novelty": novelty,
            "near_obstacle": near_obstacle,
        }

    def end_episode(self, success: bool, episode_reward: float = 0.0):
        """Episode 結束時更新統計。"""
        self._success_history.append(1.0 if success else 0.0)
        self._episode_rewards.append(episode_reward)
        self._success_rate = (
            sum(self._success_history) / len(self._success_history)
            if self._success_history else 0.0
        )
        # Reset novelty for next episode
        self._visited_cells.clear()

    def _compute_novelty(self, pos: np.ndarray) -> float:
        """
        Grid-based state novelty.
        
        把位置離散化到 grid cell，如果是新 cell → novelty = 1.0
        Phase 2 可改為更精細的 RND 或 count-based 方法。
        """
        cell = (
            int(pos[0] / self.novelty_grid_size),
            int(pos[1] / self.novelty_grid_size),
        )
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            return 1.0
        return 0.0

    def _compute_performance_trend(self) -> float:
        """計算近期效能趨勢 (正=改善中, 負=退步中)。"""
        if len(self._episode_rewards) < 10:
            return 0.0
        rewards = list(self._episode_rewards)
        recent = np.mean(rewards[-10:])
        older = np.mean(rewards[-20:-10]) if len(rewards) >= 20 else np.mean(rewards)
        return recent - older

    @property
    def success_rate(self) -> float:
        return self._success_rate

    def get_stats(self) -> Dict:
        return {
            "success_rate": self._success_rate,
            "visited_cells": len(self._visited_cells),
            "episodes_tracked": len(self._success_history),
        }

    def reset(self):
        """完全重置（新的訓練開始）。"""
        self._visited_cells.clear()
        self._success_history.clear()
        self._episode_rewards.clear()
        self._success_rate = 0.0
