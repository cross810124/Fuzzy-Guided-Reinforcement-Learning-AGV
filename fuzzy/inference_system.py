"""
Fuzzy Inference System (FIS) — Main Engine.
============================================

完整的 Mamdani 式模糊推論流程：
    1. Fuzzification:  crisp inputs → membership degrees
    2. Rule Evaluation: 計算各規則觸發強度
    3. Aggregation:     加權聚合同一輸出維度的規則
    4. Output:          4 維 guidance signal [0, 1]

4 個 Guidance 輸出維度：
    - safety_level:              安全等級 (高=安全)
    - exploration_encouragement: 探索鼓勵 (高=鼓勵探索)
    - action_confidence:         動作信心度 (高=信心高)
    - speed_modifier:            速度修正 (高=可加速)

Usage:
    from fuzzy.inference_system import FuzzyInferenceSystem
    
    fis = FuzzyInferenceSystem()
    
    guidance = fis.infer(
        obstacle_distance=1.2,
        goal_distance=5.0,
        velocity=0.5,
        goal_angle=0.3,
    )
    # guidance = {
    #     "safety_level": 0.42,
    #     "exploration_encouragement": 0.65,
    #     "action_confidence": 0.78,
    #     "speed_modifier": 0.55,
    # }
    
    # Or from observation vector directly:
    guidance = fis.infer_from_obs(obs)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass

import numpy as np

from fuzzy.membership_functions import MembershipFunctions
from fuzzy.rule_base import FuzzyRuleBase, FuzzyRule


# Guidance output dimensions
GUIDANCE_DIMS = [
    "safety_level",
    "exploration_encouragement",
    "action_confidence",
    "speed_modifier",
]

# Default values when no rule fires for a dimension
DEFAULT_GUIDANCE = {
    "safety_level": 0.5,
    "exploration_encouragement": 0.5,
    "action_confidence": 0.5,
    "speed_modifier": 0.5,
}


@dataclass
class FuzzyGuidance:
    """模糊推論結果。"""
    safety_level: float = 0.5
    exploration_encouragement: float = 0.5
    action_confidence: float = 0.5
    speed_modifier: float = 0.5
    
    # Debug info
    active_rules: int = 0
    rule_details: Optional[List[Dict]] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "safety_level": self.safety_level,
            "exploration_encouragement": self.exploration_encouragement,
            "action_confidence": self.action_confidence,
            "speed_modifier": self.speed_modifier,
        }

    def to_array(self) -> np.ndarray:
        """轉為 numpy array [safety, exploration, confidence, speed]。"""
        return np.array([
            self.safety_level,
            self.exploration_encouragement,
            self.action_confidence,
            self.speed_modifier,
        ], dtype=np.float32)

    def __repr__(self):
        return (f"FuzzyGuidance(safety={self.safety_level:.3f}, "
                f"exploration={self.exploration_encouragement:.3f}, "
                f"confidence={self.action_confidence:.3f}, "
                f"speed={self.speed_modifier:.3f}, "
                f"rules={self.active_rules})")


class FuzzyInferenceSystem:
    """
    完整模糊推論系統。
    
    Pipeline: crisp inputs → fuzzify → rule evaluation → aggregation → guidance
    
    Args:
        rule_base: 自訂規則庫 (None = 使用 Phase 1 預設 15 條規則)
        debug: 是否保留規則觸發細節
    """

    def __init__(self, rule_base: Optional[FuzzyRuleBase] = None, debug: bool = False):
        self.mf = MembershipFunctions()
        self.rule_base = rule_base or FuzzyRuleBase()
        self.debug = debug

    def infer(
        self,
        obstacle_distance: float,
        goal_distance: float,
        velocity: float,
        goal_angle: float,
    ) -> FuzzyGuidance:
        """
        執行完整模糊推論。
        
        Args:
            obstacle_distance: LiDAR 測距 [0, 10] m
            goal_distance:     到目標距離 [0, 15] m
            velocity:          線速度 (abs) [0, 2] m/s
            goal_angle:        目標相對角度 [-π, π] rad
            
        Returns:
            FuzzyGuidance with 4 guidance signals in [0, 1]
        """
        # ---- Step 1: Fuzzification ----
        fuzzified = self.mf.fuzzify_all(
            obstacle_distance=obstacle_distance,
            goal_distance=goal_distance,
            velocity=velocity,
            goal_angle=goal_angle,
        )

        # ---- Step 2: Rule Evaluation ----
        fired_rules = self.rule_base.evaluate_all(fuzzified)

        # ---- Step 3: Weighted Aggregation (Defuzzification) ----
        guidance_values = self._aggregate(fired_rules)

        # ---- Step 4: Build result ----
        result = FuzzyGuidance(
            safety_level=guidance_values["safety_level"],
            exploration_encouragement=guidance_values["exploration_encouragement"],
            action_confidence=guidance_values["action_confidence"],
            speed_modifier=guidance_values["speed_modifier"],
            active_rules=len(fired_rules),
            rule_details=fired_rules if self.debug else None,
        )

        return result

    def infer_from_obs(self, obs: np.ndarray) -> FuzzyGuidance:
        """
        從 NavigationEnv 的 observation vector 直接推論。
        
        Observation Space (9-dim):
            [0] pos_x, [1] pos_y, [2] orientation,
            [3] linear_vel, [4] angular_vel,
            [5] goal_x, [6] goal_y,
            [7] lidar_distance, [8] goal_angle
        """
        # Extract inputs from observation
        pos = obs[:2]
        goal = obs[5:7]

        obstacle_distance = float(obs[7])                         # lidar
        goal_distance = float(np.linalg.norm(pos - goal))        # computed
        velocity = float(abs(obs[3]))                             # |linear_vel|
        goal_angle = float(obs[8])                                # goal_angle

        return self.infer(
            obstacle_distance=obstacle_distance,
            goal_distance=goal_distance,
            velocity=velocity,
            goal_angle=goal_angle,
        )

    def _aggregate(self, fired_rules: List[Dict]) -> Dict[str, float]:
        """
        加權聚合（論文公式）：
        
            Output_dim = Σ(w_i × c_i) / Σ(w_i)
        
        其中 w_i = firing_strength, c_i = output crisp value
        """
        # Accumulators per output dimension
        weighted_sum = {dim: 0.0 for dim in GUIDANCE_DIMS}
        weight_sum = {dim: 0.0 for dim in GUIDANCE_DIMS}

        for rule_info in fired_rules:
            strength = rule_info["firing_strength"]
            outputs = rule_info["outputs"]

            for dim, value in outputs.items():
                if dim in weighted_sum:
                    weighted_sum[dim] += strength * value
                    weight_sum[dim] += strength

        # Compute final values with defaults for unfired dimensions
        result = {}
        for dim in GUIDANCE_DIMS:
            if weight_sum[dim] > 0:
                result[dim] = weighted_sum[dim] / weight_sum[dim]
            else:
                result[dim] = DEFAULT_GUIDANCE[dim]

        # Speed modifier: ensure minimum 0.1 (論文: max(0.1, ...))
        result["speed_modifier"] = max(0.1, result["speed_modifier"])

        # Clip all to [0, 1]
        for dim in GUIDANCE_DIMS:
            result[dim] = float(np.clip(result[dim], 0.0, 1.0))

        return result

    def get_info(self) -> Dict:
        """系統資訊。"""
        return {
            "total_rules": len(self.rule_base),
            "input_variables": self.mf.variable_names,
            "output_dimensions": GUIDANCE_DIMS,
            "rule_base": repr(self.rule_base),
        }
