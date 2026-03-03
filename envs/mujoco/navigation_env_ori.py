"""
MuJoCo Navigation Environment for Fuzzy-Guided RL AGV
======================================================
Gymnasium-compatible environment for differential-drive AGV navigation.

Observation Space (9-dim):
    [pos_x, pos_y, orientation, linear_vel, angular_vel,
     goal_x, goal_y, lidar_distance, goal_angle]

Action Space (2-dim continuous):
    [left_wheel_vel, right_wheel_vel]  normalized to [-1, 1]

Control Interface:
    Left/Right wheel velocity (differential drive)

Sensor:
    1-ray front-facing LiDAR (rangefinder, max 10m)
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class NavigationEnv(gym.Env):
    """Simple navigation environment: drive to goal in empty arena."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        arena_size: float = 2.0,        # 縮小場地（goal距離 1.5~4m）
        goal_threshold: float = 0.3,    # success distance
        max_wheel_speed: float = 10.0,  # 對齊 actuator ctrl range [-10, 10]
        sim_steps_per_action: int = 5,  # action repeat (50ms per action @ 10ms timestep)
    ):
        super().__init__()

        # --- Load MuJoCo model ---
        model_path = Path(__file__).parent / "assets" / "ammr_simple.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # --- Environment parameters ---
        self.max_steps = max_steps
        self.arena_size = arena_size
        self.goal_threshold = goal_threshold
        self.max_wheel_speed = max_wheel_speed
        self.sim_steps_per_action = sim_steps_per_action
        self.render_mode = render_mode

        # --- Track width for kinematics ---
        self.track_width = 0.32
        self.wheel_radius = 0.04

        # --- State tracking ---
        self.step_count = 0
        self.goal_pos = np.zeros(2)
        self.prev_goal_dist = 0.0

        # --- Gymnasium spaces ---
        obs_low = np.array([
            -6, -6, -np.pi, -3.0, -20.0,
            -6, -6, 0.0, -np.pi,
        ], dtype=np.float32)

        obs_high = np.array([
            6, 6, np.pi, 3.0, 20.0,
            6, 6, 10.0, np.pi,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action: [left_wheel, right_wheel] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- Renderer ---
        self.renderer = None
        if render_mode == "human" or render_mode == "rgb_array":
            self._setup_renderer()

        # --- Cache sensor addresses ---
        self._cache_sensor_ids()

    def _cache_sensor_ids(self):
        """Cache sensor data addresses for fast access."""
        self._sensor_adr = {}
        self._sensor_dim = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sensor_adr[name] = self.model.sensor_adr[i]
            self._sensor_dim[name] = self.model.sensor_dim[i]

    def _get_sensor(self, name: str) -> np.ndarray:
        """Get sensor data by name."""
        adr = self._sensor_adr[name]
        dim = self._sensor_dim[name]
        return self.data.sensordata[adr:adr + dim].copy()

    def _setup_renderer(self):
        """Setup MuJoCo renderer."""
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    # ================================================================
    # Core Gymnasium API
    # ================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Random start position
        start_x = self.np_random.uniform(-self.arena_size + 0.5, self.arena_size - 0.5)
        start_y = self.np_random.uniform(-self.arena_size + 0.5, self.arena_size - 0.5)
        start_yaw = self.np_random.uniform(-np.pi, np.pi)

        # Set initial pose (freejoint: [x, y, z, qw, qx, qy, qz])
        self.data.qpos[0] = start_x
        self.data.qpos[1] = start_y
        self.data.qpos[2] = 0.06
        self.data.qpos[3] = np.cos(start_yaw / 2)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = np.sin(start_yaw / 2)

        # Random goal (at least 1.0m from start)
        for _ in range(100):
            gx = self.np_random.uniform(-self.arena_size + 0.3, self.arena_size - 0.3)
            gy = self.np_random.uniform(-self.arena_size + 0.3, self.arena_size - 0.3)
            if np.sqrt((gx - start_x)**2 + (gy - start_y)**2) > 1.0:
                break
        self.goal_pos = np.array([gx, gy])

        # Move goal marker visual
        goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
        )
        self.model.geom_pos[goal_geom_id] = [gx, gy, 0.005]

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_goal_dist = self._goal_distance()

        return self._get_obs(), {"goal_pos": self.goal_pos.copy()}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        # Scale action [-1, 1] → actuator range
        ctrl = np.clip(action, -1.0, 1.0) * self.max_wheel_speed
        self.data.ctrl[0] = ctrl[0]  # left wheel
        self.data.ctrl[1] = ctrl[1]  # right wheel

        # Step simulation
        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)

        terminated = False
        truncated = False
        info = {"reward_info": reward_info}

        goal_dist = self._goal_distance()

        # Success
        if goal_dist < self.goal_threshold:
            terminated = True
            reward += 100.0   # 加大成功獎勵
            info["success"] = True

        # Collision
        if self._check_collision():
            terminated = True
            reward -= 20.0    # 加大碰撞懲罰
            info["collision"] = True

        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True
            info["timeout"] = True

        self.prev_goal_dist = goal_dist

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data)
        if self.render_mode == "rgb_array":
            return self.renderer.render()
        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    # ================================================================
    # Observation & Reward
    # ================================================================

    def _get_obs(self) -> np.ndarray:
        """Build 9-dim observation vector."""
        pos = self._get_sensor("chassis_pos")
        pos_x, pos_y = pos[0], pos[1]

        quat = self._get_sensor("chassis_quat")
        yaw = self._quat_to_yaw(quat)

        linvel = self._get_sensor("chassis_linvel")
        angvel = self._get_sensor("chassis_angvel")

        # Linear velocity in robot frame (forward direction)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        linear_vel = cos_yaw * linvel[0] + sin_yaw * linvel[1]
        angular_vel = angvel[2]

        # LiDAR (-1 means no hit → use max range)
        lidar_raw = self._get_sensor("lidar_front")[0]
        lidar_dist = lidar_raw if lidar_raw >= 0 else 10.0

        # Goal angle relative to robot heading
        dx = self.goal_pos[0] - pos_x
        dy = self.goal_pos[1] - pos_y
        goal_angle = self._normalize_angle(np.arctan2(dy, dx) - yaw)

        return np.array([
            pos_x, pos_y, yaw,
            linear_vel, angular_vel,
            self.goal_pos[0], self.goal_pos[1],
            lidar_dist, goal_angle,
        ], dtype=np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Base reward (Phase 1 v4 — 修正轉圈問題).
        
        核心思路：只獎勵「實際靠近目標」，不單獨獎勵 heading。
        heading 資訊透過 progress 自然傳遞（面朝目標前進 → 距離縮短 → 拿到獎勵）。
        """
        goal_dist = self._goal_distance()
        obs = self._get_obs()
        linear_vel = obs[3]

        # 1. Progress: 唯一的主要正向獎勵信號
        #    靠近目標 → 正，遠離 → 負，原地不動 → 0
        r_progress = (self.prev_goal_dist - goal_dist) * 30.0

        # 2. Alive penalty: 每步小懲罰，逼 agent 盡快到達
        r_time = -0.02

        # 3. Speed bonus: 有在移動就給小獎勵（避免原地不動）
        #    但很小，不會主導行為
        r_speed = 0.005 * min(abs(linear_vel), 0.5)

        total = r_progress + r_time + r_speed

        info = {
            "r_progress": r_progress,
            "r_time": r_time,
            "r_speed": r_speed,
            "goal_dist": goal_dist,
        }
        return total, info

    # ================================================================
    # Helpers
    # ================================================================

    def _goal_distance(self) -> float:
        pos = self._get_sensor("chassis_pos")[:2]
        return float(np.linalg.norm(pos - self.goal_pos))

    def _check_collision(self) -> bool:
        wall_names = {"wall_north", "wall_south", "wall_east", "wall_west"}
        robot_names = {"chassis_geom", "left_wheel_geom", "right_wheel_geom",
                       "front_caster_geom", "rear_caster_geom"}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if g1 and g2:
                names = {g1, g2}
                if names & wall_names and names & robot_names:
                    return True
        return False

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    # ================================================================
    # Properties
    # ================================================================

    @property
    def robot_position(self) -> np.ndarray:
        return self._get_sensor("chassis_pos")[:2].copy()

    @property
    def robot_yaw(self) -> float:
        return self._quat_to_yaw(self._get_sensor("chassis_quat"))

    @property
    def lidar_reading(self) -> float:
        val = self._get_sensor("lidar_front")[0]
        return float(val if val >= 0 else 10.0)