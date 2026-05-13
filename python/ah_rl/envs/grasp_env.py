"""Gymnasium environment for UR10e + Ability Hand object grasping."""

import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


# Default path to the grasp scene XML (relative to this file)
_DEFAULT_SCENE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "ah_simulators",
    "mujoco_xml",
    "universal_robots_ur10e",
    "grasp_scene.xml",
)


class AHGraspEnv(gym.Env):
    """UR10e + Ability Hand grasping environment.

    The agent controls the 6 arm joints and 6 hand DOFs (4 finger MCPs +
    thumb flexor + thumb rotator) to reach, grasp, and lift a box resting
    on a table.  PIP joints are auto-mimicked via the 4-bar linkage formula.

    Observation (70-dim):
        - arm joint positions (6)
        - arm joint velocities (6)
        - hand joint positions (6)
        - hand joint velocities (6)
        - FSR touch sensors (30)
        - object position (3)
        - object orientation quaternion (4)
        - object velocity (6)
        - palm-to-object relative position (3)

    Action (12-dim, continuous [-1, 1]):
        Normalized target positions for each controlled joint, scaled to
        the joint limits.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Ctrl indices for the 12 controlled DOFs
    # arm: 0-5, hand MCPs: 6,8,10,12, thumb flexor: 14, thumb rotator: 15
    _CTRL_IDS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]

    # Matching qpos indices for the 6 hand DOFs we control
    _HAND_QPOS_IDS = [6, 8, 10, 12, 15, 14]  # MCP indices + thumb_mcp + thumb_cmc

    # PIP actuator indices and their corresponding MCP actuator indices
    _PIP_CTRL_IDS = [7, 9, 11, 13]
    _MCP_CTRL_IDS = [6, 8, 10, 12]

    # 4-bar linkage mimic coefficients
    _MIMIC_SLOPE = 1.05851325
    _MIMIC_INTERCEPT = 0.72349796

    def __init__(
        self,
        scene_path: Optional[str] = None,
        frame_skip: int = 10,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        obj_pos_range: Tuple[float, float] = (0.1, 0.1),
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            scene_path: Path to MuJoCo XML. Uses default grasp scene if None.
            frame_skip: Number of mj_step calls per env step.
            max_episode_steps: Maximum steps before truncation.
            render_mode: "human" for viewer, "rgb_array" for pixel obs.
            obj_pos_range: (dx, dy) range for randomizing object position
                on the table around its default position.
            reward_weights: Override default reward component weights.
        """
        super().__init__()

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.obj_pos_range = obj_pos_range

        # Reward weights
        self.rw = {
            "reach": 1.0,
            "grasp": 0.5,
            "lift": 2.0,
            "hold": 0.5,
            "action_penalty": 0.01,
            "drop_penalty": 5.0,
        }
        if reward_weights:
            self.rw.update(reward_weights)

        # Load MuJoCo model
        scene = scene_path or os.path.normpath(_DEFAULT_SCENE)
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)

        # Cache body/joint/sensor IDs
        self._palm_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ability_hand"
        )
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object"
        )
        self._obj_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint"
        )
        self._obj_qpos_adr = self.model.jnt_qposadr[self._obj_joint_id]
        self._obj_qvel_adr = self.model.jnt_dofadr[self._obj_joint_id]

        # Palm position sensor (last 3 values in sensordata)
        self._palm_sensor_adr = self.model.sensor_adr[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_pos"
            )
        ]

        # Table top height (table body pos z + table_top geom pos z + half-height)
        self._table_z = 0.42  # table body z=0, geom pos z=0.4, half-size z=0.02

        # Default object pose (from scene)
        self._default_obj_pos = np.array([0.5, 0.0, 0.445])

        # Build ctrl limits for the 12 controlled DOFs
        ctrl_low = self.model.actuator_ctrlrange[self._CTRL_IDS, 0]
        ctrl_high = self.model.actuator_ctrlrange[self._CTRL_IDS, 1]
        self._ctrl_low = ctrl_low
        self._ctrl_high = ctrl_high
        self._ctrl_center = (ctrl_high + ctrl_low) / 2
        self._ctrl_range = (ctrl_high - ctrl_low) / 2

        # Action space: normalized [-1, 1] for each controlled DOF
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Observation space
        obs_dim = 6 + 6 + 6 + 6 + 30 + 3 + 4 + 6 + 3  # = 70
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Rendering
        self._viewer = None
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Episode state
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Reset to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Randomize object position on the table
        dx = self.np_random.uniform(-self.obj_pos_range[0], self.obj_pos_range[0])
        dy = self.np_random.uniform(-self.obj_pos_range[1], self.obj_pos_range[1])
        obj_pos = self._default_obj_pos + np.array([dx, dy, 0.0])
        self.data.qpos[self._obj_qpos_adr : self._obj_qpos_adr + 3] = obj_pos
        # Reset orientation to upright
        self.data.qpos[self._obj_qpos_adr + 3 : self._obj_qpos_adr + 7] = [
            1, 0, 0, 0,
        ]
        # Zero object velocity
        self.data.qvel[self._obj_qvel_adr : self._obj_qvel_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)

        # Map normalized actions to ctrl targets
        ctrl_targets = self._ctrl_center + action * self._ctrl_range
        for i, ctrl_id in enumerate(self._CTRL_IDS):
            self.data.ctrl[ctrl_id] = ctrl_targets[i]

        # Apply PIP mimic
        for mcp_id, pip_id in zip(self._MCP_CTRL_IDS, self._PIP_CTRL_IDS):
            self.data.ctrl[pip_id] = (
                self.data.ctrl[mcp_id] * self._MIMIC_SLOPE + self._MIMIC_INTERCEPT
            )

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self._step_count >= self.max_episode_steps

        info = self._get_info()
        info.update(reward_info)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            else:
                self._viewer.sync()
        elif self.render_mode == "rgb_array":
            self._renderer.update_scene(self.data, camera="grasp_cam")
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Arm joint positions and velocities (qpos/qvel indices 0-5)
        arm_pos = self.data.qpos[0:6].copy()
        arm_vel = self.data.qvel[0:6].copy()

        # Hand joint positions and velocities (6 controlled DOFs)
        hand_pos = self.data.qpos[self._HAND_QPOS_IDS].copy()
        hand_vel = self.data.qvel[self._HAND_QPOS_IDS].copy()

        # FSR touch sensors (first 30 sensordata values)
        fsr = self.data.sensordata[0:30].copy()

        # Object state
        obj_pos = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ].copy()
        obj_quat = self.data.qpos[
            self._obj_qpos_adr + 3 : self._obj_qpos_adr + 7
        ].copy()
        obj_vel = self.data.qvel[
            self._obj_qvel_adr : self._obj_qvel_adr + 6
        ].copy()

        # Palm-to-object relative position
        palm_pos = self.data.sensordata[
            self._palm_sensor_adr : self._palm_sensor_adr + 3
        ].copy()
        rel_pos = obj_pos - palm_pos

        return np.concatenate([
            arm_pos,     # 6
            arm_vel,     # 6
            hand_pos,    # 6
            hand_vel,    # 6
            fsr,         # 30
            obj_pos,     # 3
            obj_quat,    # 4
            obj_vel,     # 6
            rel_pos,     # 3
        ])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self, action: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        palm_pos = self.data.sensordata[
            self._palm_sensor_adr : self._palm_sensor_adr + 3
        ]
        obj_pos = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ]

        # --- Reach reward: negative distance palm -> object ---
        distance = np.linalg.norm(palm_pos - obj_pos)
        reach_reward = -distance

        # --- Grasp reward: FSR activation ---
        fsr = self.data.sensordata[0:30]
        total_fsr = np.sum(fsr)
        grasp_reward = np.tanh(total_fsr / 10.0)  # saturates at ~1.0

        # --- Lift reward: object height above table ---
        obj_z = obj_pos[2]
        lift_height = max(0.0, obj_z - self._table_z)
        lift_reward = np.tanh(lift_height / 0.05)  # saturates around 5cm lift

        # --- Hold reward: sustained grasp while lifted ---
        is_grasping = total_fsr > 1.0
        is_lifted = lift_height > 0.02
        hold_reward = 1.0 if (is_grasping and is_lifted) else 0.0

        # --- Action penalty: smooth control ---
        action_penalty = np.sum(action ** 2)

        # --- Drop penalty: object fell off table ---
        obj_fell = obj_z < (self._table_z - 0.1)
        drop_penalty = 1.0 if obj_fell else 0.0

        reward = (
            self.rw["reach"] * reach_reward
            + self.rw["grasp"] * grasp_reward
            + self.rw["lift"] * lift_reward
            + self.rw["hold"] * hold_reward
            - self.rw["action_penalty"] * action_penalty
            - self.rw["drop_penalty"] * drop_penalty
        )

        info = {
            "r_reach": reach_reward,
            "r_grasp": grasp_reward,
            "r_lift": lift_reward,
            "r_hold": hold_reward,
            "r_action_pen": -action_penalty,
            "r_drop_pen": -drop_penalty,
            "distance": distance,
            "fsr_total": total_fsr,
            "lift_height": lift_height,
        }

        return float(reward), info

    # ------------------------------------------------------------------
    # Termination & info
    # ------------------------------------------------------------------

    def _check_terminated(self) -> bool:
        obj_pos = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ]
        # Terminate if object falls far below table
        if obj_pos[2] < (self._table_z - 0.15):
            return True
        # Terminate if object flies far from table (e.g. knocked away)
        if np.linalg.norm(obj_pos[:2] - self._default_obj_pos[:2]) > 0.6:
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        palm_pos = self.data.sensordata[
            self._palm_sensor_adr : self._palm_sensor_adr + 3
        ]
        obj_pos = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ]
        return {
            "step": self._step_count,
            "palm_pos": palm_pos.copy(),
            "obj_pos": obj_pos.copy(),
            "obj_z": float(obj_pos[2]),
        }
