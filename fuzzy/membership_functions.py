"""
Membership Functions for Fuzzy Inference System.
=================================================

基於論文定義的三角形隸屬函數 (Triangular Membership Function)。

Phase 1 輸入變數 (4 個，可直接從 observation 取得):
    - obstacle_distance: LiDAR 測距 [0, 10] m
    - goal_distance:     到目標距離 [0, 15] m
    - velocity:          線速度 [0, 2] m/s
    - goal_angle:        目標相對角度 [0, π] rad (取絕對值)

Phase 2 擴充變數 (待加入):
    - environment_complexity
    - robot_battery_level
    - task_urgency
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


# ================================================================
# Triangular Membership Function
# ================================================================

@dataclass
class TriangularMF:
    """
    三角形隸屬函數 μ(x; a, b, c)
    
        μ = 0              if x <= a or x >= c
        μ = (x - a)/(b - a)  if a < x <= b
        μ = (c - x)/(c - b)  if b < x < c
    
    支援左肩 (a == b) 和右肩 (b == c) 的梯形退化。
    """
    a: float  # left foot
    b: float  # peak
    c: float  # right foot

    def __call__(self, x: float) -> float:
        x = float(x)
        if x <= self.a or x >= self.c:
            # 特殊處理: 左肩 (a==b==c 時 peak) 或右肩
            if self.a == self.b and x <= self.a:
                return 1.0
            if self.b == self.c and x >= self.c:
                return 1.0
            return 0.0
        if x <= self.b:
            if self.b == self.a:
                return 1.0
            return (x - self.a) / (self.b - self.a)
        else:
            if self.c == self.b:
                return 1.0
            return (self.c - x) / (self.c - self.b)

    def __repr__(self):
        return f"TriMF({self.a}, {self.b}, {self.c})"


# ================================================================
# Fuzzy Variable: a named variable with its linguistic terms
# ================================================================

@dataclass
class FuzzyVariable:
    """模糊變數：包含名稱、值域、和語言項 (linguistic terms)。"""
    name: str
    universe_min: float
    universe_max: float
    terms: Dict[str, TriangularMF] = field(default_factory=dict)

    def fuzzify(self, x: float) -> Dict[str, float]:
        """
        模糊化：計算 crisp input 對各語言項的隸屬度。
        
        Args:
            x: crisp input value
            
        Returns:
            dict: {term_name: membership_degree}
        """
        x = np.clip(x, self.universe_min, self.universe_max)
        return {name: mf(x) for name, mf in self.terms.items()}

    def add_term(self, name: str, a: float, b: float, c: float):
        """新增語言項。"""
        self.terms[name] = TriangularMF(a, b, c)
        return self


# ================================================================
# Phase 1 Membership Function Definitions
# ================================================================

def create_obstacle_distance_var() -> FuzzyVariable:
    """
    障礙物距離 [0, 10] m
    
    語言項（參考論文 obstacle_distance 圖）：
        VERY_NEAR:  [0, 0, 0.5]    — 碰撞危險
        NEAR:       [0.2, 1.0, 2.0] — 需注意
        MEDIUM:     [1.5, 3.0, 5.0] — 中等距離
        FAR:        [4.0, 6.0, 8.0] — 安全距離
        VERY_FAR:   [7.0, 10, 10]   — 完全安全
    """
    var = FuzzyVariable("obstacle_distance", 0.0, 10.0)
    var.add_term("VERY_NEAR", 0.0, 0.0, 0.5)
    var.add_term("NEAR",      0.2, 1.0, 2.0)
    var.add_term("MEDIUM",    1.5, 3.0, 5.0)
    var.add_term("FAR",       4.0, 6.0, 8.0)
    var.add_term("VERY_FAR",  7.0, 10.0, 10.0)
    return var


def create_goal_distance_var() -> FuzzyVariable:
    """
    目標距離 [0, 15] m
    
    語言項：
        VERY_CLOSE: [0, 0, 0.5]     — 即將到達
        CLOSE:      [0.2, 1.0, 2.5] — 接近目標
        MEDIUM:     [2.0, 4.0, 7.0] — 中等距離
        FAR:        [5.0, 8.0, 11]  — 較遠
        VERY_FAR:   [9.0, 15, 15]   — 很遠
    """
    var = FuzzyVariable("goal_distance", 0.0, 15.0)
    var.add_term("VERY_CLOSE", 0.0, 0.0, 0.5)
    var.add_term("CLOSE",      0.2, 1.0, 2.5)
    var.add_term("MEDIUM",     2.0, 4.0, 7.0)
    var.add_term("FAR",        5.0, 8.0, 11.0)
    var.add_term("VERY_FAR",   9.0, 15.0, 15.0)
    return var


def create_velocity_var() -> FuzzyVariable:
    """
    線速度（絕對值） [0, 2] m/s
    
    語言項：
        VERY_SLOW: [0, 0, 0.2]    — 幾乎靜止
        SLOW:      [0.1, 0.3, 0.6] — 慢速
        MEDIUM:    [0.4, 0.7, 1.0] — 中速
        FAST:      [0.8, 1.2, 1.6] — 快速
        VERY_FAST: [1.4, 2.0, 2.0] — 極快
    """
    var = FuzzyVariable("velocity", 0.0, 2.0)
    var.add_term("VERY_SLOW", 0.0, 0.0, 0.2)
    var.add_term("SLOW",      0.1, 0.3, 0.6)
    var.add_term("MEDIUM",    0.4, 0.7, 1.0)
    var.add_term("FAST",      0.8, 1.2, 1.6)
    var.add_term("VERY_FAST", 1.4, 2.0, 2.0)
    return var


def create_goal_angle_var() -> FuzzyVariable:
    """
    目標角度（取絕對值） [0, π] rad
    
    語言項：
        STRAIGHT:   [0, 0, 0.3]        — 正前方 (~17°)
        SLIGHT:     [0.15, 0.5, 0.9]   — 微偏 (~29°~52°)
        MODERATE:   [0.6, 1.1, 1.6]    — 中等偏轉
        LARGE:      [1.3, 2.0, 2.7]    — 大角度偏轉
        BEHIND:     [2.3, π, π]        — 目標在後方
    """
    var = FuzzyVariable("goal_angle", 0.0, np.pi)
    var.add_term("STRAIGHT", 0.0, 0.0, 0.3)
    var.add_term("SLIGHT",   0.15, 0.5, 0.9)
    var.add_term("MODERATE", 0.6, 1.1, 1.6)
    var.add_term("LARGE",    1.3, 2.0, 2.7)
    var.add_term("BEHIND",   2.3, np.pi, np.pi)
    return var


# ================================================================
# All Variables
# ================================================================

class MembershipFunctions:
    """
    所有模糊變數的集合。
    
    Usage:
        mf = MembershipFunctions()
        result = mf.fuzzify_all(obstacle_distance=1.2, goal_distance=5.0, 
                                velocity=0.5, goal_angle=0.3)
    """

    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {
            "obstacle_distance": create_obstacle_distance_var(),
            "goal_distance": create_goal_distance_var(),
            "velocity": create_velocity_var(),
            "goal_angle": create_goal_angle_var(),
        }

    def fuzzify(self, var_name: str, value: float) -> Dict[str, float]:
        """模糊化單一變數。"""
        return self.variables[var_name].fuzzify(value)

    def fuzzify_all(self, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        模糊化全部輸入變數。
        
        Args:
            obstacle_distance: float
            goal_distance: float
            velocity: float
            goal_angle: float (會自動取絕對值)
            
        Returns:
            {var_name: {term_name: membership_degree}}
        """
        # goal_angle 取絕對值
        if "goal_angle" in kwargs:
            kwargs["goal_angle"] = abs(kwargs["goal_angle"])

        return {
            name: self.variables[name].fuzzify(kwargs[name])
            for name in self.variables
            if name in kwargs
        }

    def get_variable(self, name: str) -> FuzzyVariable:
        return self.variables[name]

    @property
    def variable_names(self):
        return list(self.variables.keys())
