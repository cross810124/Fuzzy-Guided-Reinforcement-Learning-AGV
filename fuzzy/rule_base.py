"""
Fuzzy Rule Base for Phase 1 MVP.
================================

從論文 44 條規則中精選 15 條核心規則。

選取原則：
1. 只使用 Phase 1 可用的 4 個輸入變數
   (obstacle_distance, goal_distance, velocity, goal_angle)
2. 排除需要 environment_complexity, robot_battery_level, task_urgency 的規則
3. 確保 4 個輸出維度都有足夠的規則覆蓋
4. 保留最高權重的安全規則

Phase 1 精選規則 (15 條):
    Safety:      S01, S02, S03, S04 (4)
    Exploration: E01, E03, E04, E08 (4)
    Confidence:  C01, C02, C03, C05 (4)
    Speed:       V01, V02, V04      (3)

Phase 2 擴充：加入剩餘 29 條規則 + 5 條複合規則
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np


# ================================================================
# Output linguistic term → crisp value mapping
# ================================================================

OUTPUT_LEVELS = {
    # Safety Level (higher = safer)
    "CRITICAL":  0.1,
    "LOW":       0.3,
    "MEDIUM":    0.5,
    "HIGH":      0.7,
    "VERY_HIGH": 0.9,
    
    # Speed Modifier (higher = faster)
    "VERY_SLOW": 0.1,
    "SLOW":      0.3,
    "MODERATE":  0.5,   # 中速
    "FAST":      0.7,
    "VERY_FAST": 0.9,
}


@dataclass
class FuzzyRule:
    """
    單一模糊規則。
    
    Attributes:
        rule_id:     規則編號 (e.g., "S01")
        conditions:  觸發條件 {var_name: [term1, term2, ...]}
                     多個 term 之間是 OR 關係
                     多個 variable 之間是 AND 關係
        outputs:     輸出 {output_name: linguistic_level}
        weight:      規則權重 [0, 1]
        description: 規則描述
    """
    rule_id: str
    conditions: Dict[str, List[str]]  # {var: [terms]} → terms 之間 OR, vars 之間 AND
    outputs: Dict[str, str]           # {output_dim: level_name}
    weight: float = 1.0
    description: str = ""

    def evaluate(self, fuzzified: Dict[str, Dict[str, float]]) -> float:
        """
        評估規則的觸發強度 (firing strength)。
        
        使用 min (AND) 和 max (OR) 運算。
        
        Args:
            fuzzified: {var_name: {term_name: membership_degree}}
            
        Returns:
            firing_strength: [0, 1]
        """
        var_memberships = []

        for var_name, terms in self.conditions.items():
            if var_name not in fuzzified:
                return 0.0  # 缺少輸入變數，規則不觸發

            # OR: 同一變數的多個 term 取 max
            term_values = []
            for term in terms:
                val = fuzzified[var_name].get(term, 0.0)
                term_values.append(val)
            
            var_memberships.append(max(term_values) if term_values else 0.0)

        if not var_memberships:
            return 0.0

        # AND: 不同變數之間取 min
        firing_strength = min(var_memberships)
        
        return firing_strength * self.weight

    def get_output_values(self) -> Dict[str, float]:
        """將語言項輸出轉為 crisp 數值。"""
        return {
            name: OUTPUT_LEVELS.get(level, 0.5)
            for name, level in self.outputs.items()
        }


# ================================================================
# Phase 1 Core Rules (15 rules)
# ================================================================

def create_safety_rules() -> List[FuzzyRule]:
    """安全性規則 (4 條)：來自論文 S01-S04"""
    return [
        FuzzyRule(
            rule_id="S01",
            conditions={"obstacle_distance": ["VERY_NEAR"]},
            outputs={"safety_level": "CRITICAL"},
            weight=1.0,
            description="極近障礙物 → 安全等級嚴重",
        ),
        FuzzyRule(
            rule_id="S02",
            conditions={
                "obstacle_distance": ["NEAR"],
                "velocity": ["FAST", "VERY_FAST"],
            },
            outputs={"safety_level": "LOW"},
            weight=0.9,
            description="近距離 + 高速 → 安全性低",
        ),
        FuzzyRule(
            rule_id="S03",
            conditions={
                "obstacle_distance": ["NEAR"],
                "velocity": ["VERY_SLOW", "SLOW"],
            },
            outputs={"safety_level": "MEDIUM"},
            weight=0.8,
            description="近距離 + 低速 → 安全性中等",
        ),
        FuzzyRule(
            rule_id="S04",
            conditions={"obstacle_distance": ["FAR"]},
            outputs={"safety_level": "HIGH"},
            weight=0.7,
            description="遠離障礙物 → 安全性高",
        ),
    ]


def create_exploration_rules() -> List[FuzzyRule]:
    """探索規則 (4 條)：來自論文 E01, E03, E04, E08"""
    return [
        FuzzyRule(
            rule_id="E01",
            conditions={
                "goal_distance": ["FAR", "VERY_FAR"],
                "obstacle_distance": ["FAR", "VERY_FAR"],
            },
            outputs={"exploration_encouragement": "HIGH"},
            weight=0.8,
            description="遠離目標 + 開闊 → 鼓勵探索",
        ),
        FuzzyRule(
            rule_id="E03",
            conditions={"obstacle_distance": ["VERY_NEAR"]},
            outputs={"exploration_encouragement": "VERY_SLOW"},  # VERY_LOW → 0.1
            weight=1.0,
            description="接近障礙物 → 不鼓勵探索",
        ),
        FuzzyRule(
            rule_id="E04",
            conditions={
                "goal_distance": ["VERY_CLOSE", "CLOSE"],
                "obstacle_distance": ["MEDIUM"],
            },
            outputs={"exploration_encouragement": "LOW"},
            weight=0.7,
            description="接近目標 → 減少探索",
        ),
        FuzzyRule(
            rule_id="E08",
            conditions={
                "velocity": ["VERY_SLOW"],
                "obstacle_distance": ["FAR", "VERY_FAR"],
            },
            outputs={"exploration_encouragement": "HIGH"},
            weight=0.7,
            description="慢速 + 安全 → 可以探索",
        ),
    ]


def create_confidence_rules() -> List[FuzzyRule]:
    """信心度規則 (4 條)：來自論文 C01, C02, C03, C05"""
    return [
        FuzzyRule(
            rule_id="C01",
            conditions={
                "goal_angle": ["STRAIGHT"],
                "obstacle_distance": ["FAR", "VERY_FAR"],
            },
            outputs={"action_confidence": "HIGH"},
            weight=0.9,
            description="直行 + 無障礙 → 高信心度",
        ),
        FuzzyRule(
            rule_id="C02",
            conditions={"goal_angle": ["LARGE", "BEHIND"]},
            outputs={"action_confidence": "MEDIUM"},
            weight=0.7,
            description="大角度轉彎 → 中等信心度",
        ),
        FuzzyRule(
            rule_id="C03",
            conditions={
                "obstacle_distance": ["NEAR"],
                "goal_angle": ["SLIGHT", "MODERATE", "LARGE", "BEHIND"],
            },
            outputs={"action_confidence": "LOW"},
            weight=0.8,
            description="近障礙物 + 需轉彎 → 低信心度",
        ),
        FuzzyRule(
            rule_id="C05",
            conditions={
                "goal_distance": ["VERY_CLOSE"],
                "goal_angle": ["STRAIGHT"],
            },
            outputs={"action_confidence": "HIGH"},
            weight=0.8,
            description="直線接近目標 → 高信心度",
        ),
    ]


def create_speed_rules() -> List[FuzzyRule]:
    """速度調節規則 (3 條)：來自論文 V01, V02, V04"""
    return [
        FuzzyRule(
            rule_id="V01",
            conditions={"obstacle_distance": ["VERY_NEAR"]},
            outputs={"speed_modifier": "VERY_SLOW"},
            weight=1.0,
            description="極近障礙物 → 極慢速度",
        ),
        FuzzyRule(
            rule_id="V02",
            conditions={"obstacle_distance": ["NEAR"]},
            outputs={"speed_modifier": "SLOW"},
            weight=0.9,
            description="近障礙物 → 慢速",
        ),
        FuzzyRule(
            rule_id="V04",
            conditions={"goal_distance": ["VERY_CLOSE"]},
            outputs={"speed_modifier": "SLOW"},
            weight=0.9,
            description="接近目標 → 減速精確控制",
        ),
    ]


# ================================================================
# Create all rules
# ================================================================

class FuzzyRuleBase:
    """
    模糊規則庫。
    
    Usage:
        rb = FuzzyRuleBase()
        print(f"Total rules: {len(rb)}")
        print(f"Safety rules: {rb.get_rules_by_category('S')}")
    """

    def __init__(self, rules: Optional[List[FuzzyRule]] = None):
        if rules is not None:
            self.rules = rules
        else:
            self.rules = self._create_phase1_rules()
        
        # Build index by ID
        self._index = {r.rule_id: r for r in self.rules}

    def _create_phase1_rules(self) -> List[FuzzyRule]:
        """建立 Phase 1 核心規則 (15 條)。"""
        rules = []
        rules.extend(create_safety_rules())
        rules.extend(create_exploration_rules())
        rules.extend(create_confidence_rules())
        rules.extend(create_speed_rules())
        return rules

    def evaluate_all(self, fuzzified: Dict[str, Dict[str, float]]) -> List[Dict]:
        """
        評估所有規則。
        
        Returns:
            list of {rule_id, firing_strength, outputs}
        """
        results = []
        for rule in self.rules:
            strength = rule.evaluate(fuzzified)
            if strength > 0:
                results.append({
                    "rule_id": rule.rule_id,
                    "firing_strength": strength,
                    "outputs": rule.get_output_values(),
                })
        return results

    def get_rule(self, rule_id: str) -> Optional[FuzzyRule]:
        return self._index.get(rule_id)

    def get_rules_by_category(self, prefix: str) -> List[FuzzyRule]:
        """取得某類規則 (e.g., 'S' for Safety)。"""
        return [r for r in self.rules if r.rule_id.startswith(prefix)]

    def add_rule(self, rule: FuzzyRule):
        """動態新增規則。"""
        self.rules.append(rule)
        self._index[rule.rule_id] = rule

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        cats = {}
        for r in self.rules:
            cat = r.rule_id[0]
            cats[cat] = cats.get(cat, 0) + 1
        cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(cats.items()))
        return f"FuzzyRuleBase({len(self.rules)} rules: {cat_str})"
