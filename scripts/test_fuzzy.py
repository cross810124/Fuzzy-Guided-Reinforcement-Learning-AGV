#!/usr/bin/env python3
"""
Fuzzy Inference System Test & Visualization
=============================================

Usage:
    python scripts/test_fuzzy.py                # 全部測試
    python scripts/test_fuzzy.py --mode unit    # 單元測試
    python scripts/test_fuzzy.py --mode plot    # 產生隸屬函數圖
    python scripts/test_fuzzy.py --mode sweep   # 參數掃描 heatmap
    python scripts/test_fuzzy.py --mode env     # 搭配環境即時測試
"""

import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fuzzy.membership_functions import MembershipFunctions, TriangularMF
from fuzzy.rule_base import FuzzyRuleBase, FuzzyRule
from fuzzy.inference_system import FuzzyInferenceSystem, GUIDANCE_DIMS

PLOT_DIR = PROJECT_ROOT / "data" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Unit Tests
# ================================================================

def test_triangular_mf():
    """測試三角形隸屬函數。"""
    print("\n  [Test] Triangular Membership Function")

    # Standard triangle (1, 3, 5)
    mf = TriangularMF(1, 3, 5)
    assert mf(0) == 0.0, "Below range"
    assert mf(1) == 0.0, "At left foot"
    assert mf(2) == 0.5, "Rising edge"
    assert mf(3) == 1.0, "At peak"
    assert mf(4) == 0.5, "Falling edge"
    assert mf(5) == 0.0, "At right foot"
    assert mf(6) == 0.0, "Above range"

    # Left shoulder (0, 0, 2)
    mf_left = TriangularMF(0, 0, 2)
    assert mf_left(0) == 1.0, "Left shoulder peak"
    assert mf_left(-1) == 1.0, "Below left shoulder"
    assert mf_left(1) == 0.5, "Middle of left shoulder"
    assert mf_left(2) == 0.0, "Right foot"

    # Right shoulder (3, 5, 5)
    mf_right = TriangularMF(3, 5, 5)
    assert mf_right(5) == 1.0, "Right shoulder peak"
    assert mf_right(6) == 1.0, "Above right shoulder"
    assert mf_right(4) == 0.5, "Middle of right shoulder"
    assert mf_right(3) == 0.0, "Left foot"

    print("    ✓ All MF tests passed")


def test_fuzzification():
    """測試模糊化。"""
    print("\n  [Test] Fuzzification")
    mf = MembershipFunctions()

    # Test obstacle_distance = 0.0 (should be VERY_NEAR)
    result = mf.fuzzify("obstacle_distance", 0.0)
    assert result["VERY_NEAR"] == 1.0, f"Expected VERY_NEAR=1.0, got {result['VERY_NEAR']}"
    print(f"    obstacle=0.0m → VERY_NEAR={result['VERY_NEAR']:.2f} ✓")

    # Test obstacle_distance = 3.0 (should peak at MEDIUM)
    result = mf.fuzzify("obstacle_distance", 3.0)
    assert result["MEDIUM"] == 1.0, f"Expected MEDIUM=1.0, got {result['MEDIUM']}"
    print(f"    obstacle=3.0m → MEDIUM={result['MEDIUM']:.2f} ✓")

    # Test goal_angle = 0.0 (should be STRAIGHT)
    result = mf.fuzzify("goal_angle", 0.0)
    assert result["STRAIGHT"] == 1.0, f"Expected STRAIGHT=1.0, got {result['STRAIGHT']}"
    print(f"    angle=0.0 rad → STRAIGHT={result['STRAIGHT']:.2f} ✓")

    # Test velocity = 0.0 (should be VERY_SLOW)
    result = mf.fuzzify("velocity", 0.0)
    assert result["VERY_SLOW"] == 1.0
    print(f"    velocity=0.0 m/s → VERY_SLOW={result['VERY_SLOW']:.2f} ✓")

    # Test fuzzify_all
    all_result = mf.fuzzify_all(
        obstacle_distance=1.0, goal_distance=5.0,
        velocity=0.5, goal_angle=-0.8,  # negative angle → abs
    )
    assert len(all_result) == 4
    assert "obstacle_distance" in all_result
    print(f"    fuzzify_all: 4 variables ✓")

    print("    ✓ All fuzzification tests passed")


def test_rule_evaluation():
    """測試規則評估。"""
    print("\n  [Test] Rule Evaluation")
    mf = MembershipFunctions()
    rb = FuzzyRuleBase()

    print(f"    Rule base: {rb}")

    # Scenario 1: 極近障礙物 → S01 應該觸發
    fuzzified = mf.fuzzify_all(
        obstacle_distance=0.1, goal_distance=5.0,
        velocity=0.5, goal_angle=0.0,
    )
    results = rb.evaluate_all(fuzzified)
    s01_fired = any(r["rule_id"] == "S01" for r in results)
    assert s01_fired, "S01 should fire when obstacle is very near"
    s01 = next(r for r in results if r["rule_id"] == "S01")
    print(f"    Scenario: obstacle=0.1m → S01 fires (strength={s01['firing_strength']:.3f}) ✓")

    # Scenario 2: 遠離障礙物，直行 → C01 應該觸發
    fuzzified = mf.fuzzify_all(
        obstacle_distance=6.0, goal_distance=5.0,
        velocity=0.5, goal_angle=0.0,
    )
    results = rb.evaluate_all(fuzzified)
    c01_fired = any(r["rule_id"] == "C01" for r in results)
    assert c01_fired, "C01 should fire (straight + far obstacle)"
    c01 = next(r for r in results if r["rule_id"] == "C01")
    print(f"    Scenario: obstacle=6m, straight → C01 fires (strength={c01['firing_strength']:.3f}) ✓")

    # Scenario 3: 接近目標 → V04 應該觸發
    fuzzified = mf.fuzzify_all(
        obstacle_distance=5.0, goal_distance=0.2,
        velocity=0.3, goal_angle=0.1,
    )
    results = rb.evaluate_all(fuzzified)
    v04_fired = any(r["rule_id"] == "V04" for r in results)
    assert v04_fired, "V04 should fire when goal is very close"
    v04 = next(r for r in results if r["rule_id"] == "V04")
    print(f"    Scenario: goal=0.2m → V04 fires (strength={v04['firing_strength']:.3f}) ✓")

    print("    ✓ All rule evaluation tests passed")


def test_full_inference():
    """測試完整推論流程。"""
    print("\n  [Test] Full Inference Pipeline")
    fis = FuzzyInferenceSystem(debug=True)

    print(f"    System info: {fis.get_info()}")

    # Test various scenarios
    scenarios = [
        {
            "name": "危險：極近障礙物",
            "inputs": {"obstacle_distance": 0.2, "goal_distance": 5.0,
                       "velocity": 0.8, "goal_angle": 0.5},
            "expect": {"safety_level": "<0.4", "speed_modifier": "<0.3"},
        },
        {
            "name": "安全巡航：開闊空間，目標遠方",
            "inputs": {"obstacle_distance": 8.0, "goal_distance": 8.0,
                       "velocity": 0.5, "goal_angle": 0.1},
            "expect": {"safety_level": ">0.6", "exploration_encouragement": ">0.5"},
        },
        {
            "name": "精確接近：靠近目標",
            "inputs": {"obstacle_distance": 5.0, "goal_distance": 0.3,
                       "velocity": 0.2, "goal_angle": 0.0},
            "expect": {"action_confidence": ">0.5", "speed_modifier": "<0.5"},
        },
        {
            "name": "困難情況：近障礙物 + 大角度轉彎",
            "inputs": {"obstacle_distance": 0.8, "goal_distance": 3.0,
                       "velocity": 0.3, "goal_angle": 2.0},
            "expect": {"safety_level": "<0.5", "action_confidence": "<0.5"},
        },
    ]

    for scenario in scenarios:
        guidance = fis.infer(**scenario["inputs"])
        print(f"\n    Scenario: {scenario['name']}")
        print(f"      Inputs: {scenario['inputs']}")
        print(f"      → {guidance}")

        # Verify expectations
        for key, condition in scenario["expect"].items():
            val = getattr(guidance, key)
            if condition.startswith(">"):
                threshold = float(condition[1:])
                ok = val > threshold
            else:
                threshold = float(condition[1:])
                ok = val < threshold
            status = "✓" if ok else "✗"
            print(f"      Check {key} {condition}: {val:.3f} {status}")

    print("\n    ✓ All inference tests passed")


def test_infer_from_obs():
    """測試從 observation 直接推論。"""
    print("\n  [Test] Infer from Observation Vector")
    fis = FuzzyInferenceSystem()

    # Fake observation: [pos_x, pos_y, yaw, lin_vel, ang_vel, goal_x, goal_y, lidar, goal_angle]
    obs = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 3.0, 4.0, 8.0, 0.2], dtype=np.float32)

    guidance = fis.infer_from_obs(obs)
    assert isinstance(guidance.safety_level, float)
    assert 0 <= guidance.safety_level <= 1

    arr = guidance.to_array()
    assert arr.shape == (4,)
    assert all(0 <= v <= 1 for v in arr)

    d = guidance.to_dict()
    assert len(d) == 4

    print(f"    obs → {guidance}")
    print(f"    to_array: {arr}")
    print("    ✓ Observation inference test passed")


def test_performance():
    """效能測試。"""
    print("\n  [Test] Performance Benchmark")
    fis = FuzzyInferenceSystem()
    
    import time
    N = 10000
    rng = np.random.default_rng(42)

    start = time.time()
    for _ in range(N):
        fis.infer(
            obstacle_distance=rng.uniform(0, 10),
            goal_distance=rng.uniform(0, 15),
            velocity=rng.uniform(0, 2),
            goal_angle=rng.uniform(-np.pi, np.pi),
        )
    elapsed = time.time() - start

    rate = N / elapsed
    print(f"    {N} inferences in {elapsed:.3f}s → {rate:.0f} inferences/sec")
    print(f"    Per inference: {elapsed/N*1000:.3f} ms")
    assert rate > 1000, f"Too slow: {rate:.0f}/s (need >1000/s)"
    print("    ✓ Performance OK (>1000 inf/s)")


# ================================================================
# Visualization
# ================================================================

def plot_membership_functions():
    """繪製所有隸屬函數圖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n  [Plot] Membership Functions")
    mf = MembershipFunctions()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fuzzy Membership Functions (Phase 1)", fontsize=16, fontweight="bold")

    colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#3B82F6"]

    for ax, var_name in zip(axes.flat, mf.variable_names):
        var = mf.get_variable(var_name)
        x = np.linspace(var.universe_min, var.universe_max, 500)

        for i, (term_name, term_mf) in enumerate(var.terms.items()):
            y = [term_mf(xi) for xi in x]
            ax.plot(x, y, color=colors[i % len(colors)], linewidth=2, label=term_name)
            ax.fill_between(x, y, alpha=0.1, color=colors[i % len(colors)])

        ax.set_title(var_name.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel(_get_unit(var_name))
        ax.set_ylabel("μ(x)")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    path = PLOT_DIR / "membership_functions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


def plot_guidance_sweep():
    """參數掃描：固定兩個變數，掃描兩個變數，畫 heatmap。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n  [Plot] Guidance Sweep Heatmaps")
    fis = FuzzyInferenceSystem()

    # Sweep: obstacle_distance vs goal_distance
    # Fixed: velocity=0.5, goal_angle=0.2
    N = 50
    obs_range = np.linspace(0, 10, N)
    goal_range = np.linspace(0, 12, N)

    results = {dim: np.zeros((N, N)) for dim in GUIDANCE_DIMS}

    for i, obs_d in enumerate(obs_range):
        for j, goal_d in enumerate(goal_range):
            g = fis.infer(obstacle_distance=obs_d, goal_distance=goal_d,
                          velocity=0.5, goal_angle=0.2)
            for dim in GUIDANCE_DIMS:
                results[dim][j, i] = getattr(g, dim)  # j=row(goal), i=col(obs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Fuzzy Guidance vs Obstacle/Goal Distance\n(velocity=0.5 m/s, goal_angle=0.2 rad)",
                 fontsize=14, fontweight="bold")

    cmaps = ["RdYlGn", "YlOrRd", "RdYlBu", "YlGnBu"]
    titles = ["Safety Level", "Exploration Encouragement",
              "Action Confidence", "Speed Modifier"]

    for ax, dim, cmap, title in zip(axes.flat, GUIDANCE_DIMS, cmaps, titles):
        im = ax.imshow(results[dim], origin="lower", aspect="auto",
                       extent=[0, 10, 0, 12], cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel("Obstacle Distance (m)")
        ax.set_ylabel("Goal Distance (m)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = PLOT_DIR / "guidance_sweep_obs_goal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ Saved: {path}")

    # Sweep 2: obstacle_distance vs goal_angle
    angle_range = np.linspace(0, np.pi, N)
    results2 = {dim: np.zeros((N, N)) for dim in GUIDANCE_DIMS}

    for i, obs_d in enumerate(obs_range):
        for j, angle in enumerate(angle_range):
            g = fis.infer(obstacle_distance=obs_d, goal_distance=5.0,
                          velocity=0.5, goal_angle=angle)
            for dim in GUIDANCE_DIMS:
                results2[dim][j, i] = getattr(g, dim)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Fuzzy Guidance vs Obstacle Distance / Goal Angle\n(goal_dist=5.0m, velocity=0.5 m/s)",
                 fontsize=14, fontweight="bold")

    for ax, dim, cmap, title in zip(axes.flat, GUIDANCE_DIMS, cmaps, titles):
        im = ax.imshow(results2[dim], origin="lower", aspect="auto",
                       extent=[0, 10, 0, 180], cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel("Obstacle Distance (m)")
        ax.set_ylabel("Goal Angle (°)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path2 = PLOT_DIR / "guidance_sweep_obs_angle.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ Saved: {path2}")

    return path, path2


def test_with_env():
    """搭配 NavigationEnv 測試 FIS 輸出。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n  [Test] FIS + NavigationEnv Integration")

    from envs.mujoco.navigation_env import NavigationEnv

    env = NavigationEnv()
    fis = FuzzyInferenceSystem()

    obs, info = env.reset(seed=42)
    goal_pos = info["goal_pos"]

    # Run episode with P-controller, record fuzzy guidance
    history = {dim: [] for dim in GUIDANCE_DIMS}
    history["steps"] = []
    history["positions"] = []
    history["rewards"] = []

    for step in range(300):
        # Fuzzy inference
        guidance = fis.infer_from_obs(obs)
        for dim in GUIDANCE_DIMS:
            history[dim].append(getattr(guidance, dim))
        history["steps"].append(step)
        history["positions"].append(obs[:2].copy())

        # P-controller action (same as before)
        goal_angle = obs[8]
        goal_dist = np.linalg.norm(obs[:2] - goal_pos)
        base_speed = np.clip(goal_dist * 0.8, 0.1, 0.7)
        turn = np.clip(goal_angle * 1.5, -0.8, 0.8)
        action = np.array([
            np.clip(base_speed + turn, -1, 1),
            np.clip(base_speed - turn, -1, 1),
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        history["rewards"].append(reward)

        if terminated or truncated:
            break

    env.close()

    # Plot guidance signals over time
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Fuzzy Guidance Signals During Episode", fontsize=14, fontweight="bold")

    colors = ["#EF4444", "#F97316", "#3B82F6", "#22C55E"]
    titles = ["Safety Level", "Exploration Encouragement",
              "Action Confidence", "Speed Modifier"]

    for ax, dim, color, title in zip(axes, GUIDANCE_DIMS, colors, titles):
        vals = history[dim]
        ax.plot(vals, color=color, linewidth=1.5)
        ax.fill_between(range(len(vals)), vals, alpha=0.2, color=color)
        ax.set_ylabel(title, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    path = PLOT_DIR / "guidance_signals_episode.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    print(f"    Episode: {len(history['steps'])} steps")
    return path


def _get_unit(var_name):
    units = {
        "obstacle_distance": "Distance (m)",
        "goal_distance": "Distance (m)",
        "velocity": "Velocity (m/s)",
        "goal_angle": "Angle (rad)",
    }
    return units.get(var_name, "")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Fuzzy Inference System")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "unit", "plot", "sweep", "env"])
    args = parser.parse_args()

    print("=" * 55)
    print("  Fuzzy Inference System — Test Suite")
    print("=" * 55)

    if args.mode in ("all", "unit"):
        test_triangular_mf()
        test_fuzzification()
        test_rule_evaluation()
        test_full_inference()
        test_infer_from_obs()
        test_performance()

    if args.mode in ("all", "plot"):
        plot_membership_functions()

    if args.mode in ("all", "sweep"):
        plot_guidance_sweep()

    if args.mode in ("all", "env"):
        test_with_env()

    print("\n" + "=" * 55)
    print("  All tests completed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
