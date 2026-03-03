"""
Test Script for MuJoCo Navigation Environment
================================================
Usage:
    python scripts/test_env.py                    # all tests
    python scripts/test_env.py --test model       # model only
    python scripts/test_env.py --test env          # env basics
    python scripts/test_env.py --test drive        # P-control navigation
    python scripts/test_env.py --test random       # random stress test
    python scripts/test_env.py --episodes 20       # set episode count
"""

import sys, argparse, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_model():
    import mujoco
    print("=" * 60)
    print("TEST 1: MuJoCo Model Loading & Physics")
    print("=" * 60)

    model_path = PROJECT_ROOT / "envs" / "mujoco" / "assets" / "ammr_simple.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"  ✓ Model loaded: {model_path.name}")
    print(f"    Timestep: {model.opt.timestep}s")
    print(f"    Bodies: {model.nbody} | Joints: {model.njnt} | Actuators: {model.nu} | Sensors: {model.nsensor}")

    # Forward drive
    print("\n  Testing forward drive (2s)...")
    mujoco.mj_resetData(model, data)
    for _ in range(200):
        data.ctrl[0] = 5.0; data.ctrl[1] = 5.0
        mujoco.mj_step(model, data)
    pos = data.sensor("chassis_pos").data
    print(f"    Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    assert pos[0] > 0.1, "Robot should have moved forward!"
    print("  ✓ Forward drive OK")

    # Turn in place
    print("\n  Testing turn in place (2s)...")
    mujoco.mj_resetData(model, data)
    for _ in range(200):
        data.ctrl[0] = 5.0; data.ctrl[1] = -5.0
        mujoco.mj_step(model, data)
    quat = data.sensor("chassis_quat").data
    yaw = np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))
    print(f"    Yaw: {np.degrees(yaw):.1f}°")
    assert abs(yaw) > 0.5
    print("  ✓ Turn in place OK")
    print("\n  ✅ Model test PASSED\n")


def test_env():
    from envs.mujoco.navigation_env import NavigationEnv
    print("=" * 60)
    print("TEST 2: NavigationEnv Basic Functionality")
    print("=" * 60)

    env = NavigationEnv()
    print(f"  ✓ Environment created (obs={env.observation_space.shape}, act={env.action_space.shape})")

    obs, info = env.reset(seed=42)
    assert obs.shape == (9,)
    print(f"  ✓ Reset OK")

    labels = ["pos_x","pos_y","yaw","lin_v","ang_v","goal_x","goal_y","lidar","goal_∠"]
    for name, val in zip(labels, obs):
        print(f"    {name:8s}: {val:+.3f}")

    obs2, reward, term, trunc, info2 = env.step(np.array([0.5, 0.5], dtype=np.float32))
    assert obs2.shape == (9,) and np.isfinite(reward)
    print(f"  ✓ Step OK (reward={reward:.4f})")

    for i in range(10):
        o, _ = env.reset(seed=i*100)
        assert env.observation_space.contains(o), f"Reset {i}: obs out of bounds"
    print(f"  ✓ Observation bounds OK")

    pos = env.robot_position; yaw = env.robot_yaw; lidar = env.lidar_reading
    assert pos.shape == (2,) and -np.pi <= yaw <= np.pi and 0 <= lidar <= 10
    print(f"  ✓ Properties OK")
    env.close()
    print("\n  ✅ Environment test PASSED\n")


def test_drive(num_episodes=10):
    from envs.mujoco.navigation_env import NavigationEnv
    print("=" * 60)
    print(f"TEST 3: Proportional Control Navigation ({num_episodes} episodes)")
    print("=" * 60)

    env = NavigationEnv()
    successes = collisions = timeouts = 0
    all_rewards, all_steps = [], []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        start_pos = obs[:2].copy(); goal_pos = info["goal_pos"]
        ep_reward = 0

        for step in range(500):
            goal_angle = obs[8]
            goal_dist = np.sqrt((obs[5]-obs[0])**2 + (obs[6]-obs[1])**2)

            # P-controller: aggressive turn + slow when misaligned
            speed = min(0.6, max(0.2, goal_dist * 0.2))
            if abs(goal_angle) > 0.5:
                speed *= 0.3
            turn = np.clip(goal_angle * 3.0, -1.0, 1.0)
            left = np.clip(speed + turn * 0.8, -1.0, 1.0)
            right = np.clip(speed - turn * 0.8, -1.0, 1.0)

            obs, reward, term, trunc, info = env.step(np.array([left, right], dtype=np.float32))
            ep_reward += reward
            if term or trunc: break

        result = "SUCCESS ✓" if info.get("success") else ("COLLISION ✗" if info.get("collision") else "TIMEOUT")
        if info.get("success"): successes += 1
        elif info.get("collision"): collisions += 1
        else: timeouts += 1

        all_rewards.append(ep_reward); all_steps.append(step+1)
        print(f"  Ep {ep+1:3d} | ({start_pos[0]:+.1f},{start_pos[1]:+.1f}) → ({goal_pos[0]:+.1f},{goal_pos[1]:+.1f}) | "
              f"Steps: {step+1:3d} | Reward: {ep_reward:+7.1f} | {result}")

    print(f"\n  Summary:")
    print(f"    Success:   {successes}/{num_episodes} ({100*successes/num_episodes:.0f}%)")
    print(f"    Collision: {collisions}/{num_episodes}")
    print(f"    Timeout:   {timeouts}/{num_episodes}")
    print(f"    Avg reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"    Avg steps:  {np.mean(all_steps):.0f}")
    env.close()
    print(f"\n  ✅ Drive test DONE\n")


def test_random(num_episodes=10):
    from envs.mujoco.navigation_env import NavigationEnv
    print("=" * 60)
    print(f"TEST 4: Random Action Stress Test ({num_episodes} episodes)")
    print("=" * 60)

    env = NavigationEnv()
    results = {"success": 0, "collision": 0, "timeout": 0}
    all_rewards = []; total_steps = 0
    start_time = time.time()

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep*7)
        ep_reward = 0
        for step in range(500):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            ep_reward += reward
            assert np.all(np.isfinite(obs)), f"Ep {ep} Step {step}: non-finite obs!"
            assert np.isfinite(reward), f"Ep {ep} Step {step}: non-finite reward!"
            if term or trunc: break

        total_steps += step + 1; all_rewards.append(ep_reward)
        if info.get("success"): results["success"] += 1
        elif info.get("collision"): results["collision"] += 1
        else: results["timeout"] += 1

    elapsed = time.time() - start_time
    print(f"  Results: {results}")
    print(f"  Avg reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"  Total sim steps: {total_steps*5:,} in {elapsed:.1f}s ({total_steps*5/elapsed:,.0f} steps/s)")
    print(f"  All values finite: ✓")
    env.close()
    print(f"\n  ✅ Stress test PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="Test MuJoCo Navigation Environment")
    parser.add_argument("--test", default="all", choices=["all","model","env","drive","random"])
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Fuzzy-Guided RL AGV — Environment Test Suite           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    tests = {"model": test_model, "env": test_env,
             "drive": lambda: test_drive(args.episodes),
             "random": lambda: test_random(args.episodes)}

    if args.test == "all":
        passed = failed = 0
        for name, func in tests.items():
            try: func(); passed += 1
            except Exception as e:
                print(f"  ❌ {name} FAILED: {e}"); failed += 1
        print("=" * 60)
        print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
        print("=" * 60)
    else:
        tests[args.test]()

if __name__ == "__main__":
    main()
