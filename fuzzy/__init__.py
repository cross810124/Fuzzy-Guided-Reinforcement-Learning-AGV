"""
Fuzzy Inference System for Fuzzy-Guided Reinforcement Learning.

Phase 1 MVP: 15 core rules, 4 input variables, 4 guidance outputs.

Usage:
    from fuzzy import FuzzyInferenceSystem
    
    fis = FuzzyInferenceSystem()
    guidance = fis.infer(obstacle_distance=1.2, goal_distance=5.0,
                         velocity=0.5, goal_angle=0.3)
"""

from fuzzy.inference_system import FuzzyInferenceSystem, FuzzyGuidance, GUIDANCE_DIMS
from fuzzy.membership_functions import MembershipFunctions, FuzzyVariable, TriangularMF
from fuzzy.rule_base import FuzzyRuleBase, FuzzyRule

__all__ = [
    "FuzzyInferenceSystem",
    "FuzzyGuidance",
    "GUIDANCE_DIMS",
    "MembershipFunctions",
    "FuzzyVariable",
    "TriangularMF",
    "FuzzyRuleBase",
    "FuzzyRule",
]
