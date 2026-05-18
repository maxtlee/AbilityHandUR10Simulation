"""Gymnasium environment for hand-only Ability Hand grasping.

Architecture: the UR10e arm is in the scene but its joints are held at a
fixed "grasp_ready" pose by stiff position controllers (kp=5000). The arm
is invisible to the policy: it appears nowhere in the action or observation
space. The policy learns to control only the 6 hand DOFs to grasp a box
already positioned within finger reach.

This is functionally a fixed-base hand for the learning problem, but uses
the proven arm geometry to place the palm above the object without the
geometric headaches of welding the hand at arbitrary world poses.

Reward: bounded shaped (approach + gated contact + lift) + sparse sustained
lift bonus. No drop penalty -- terminate instead.
"""

import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


_DEFAULT_SCENE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "ah_simulators",
    "mujoco_xml",
    "universal_robots_ur10e",
    "grasp_scene_handonly.xml",
)


class AHGraspEnvFixed(gym.Env):
    """Hand-only Ability Hand grasping env (arm frozen, no arm in policy).

    Action (6-dim, continuous [-1, 1]):
        Normalized target positions for index/middle/ring/pinky MCPs,
        thumb flexor (thumb_mcp joint), thumb rotator (thumb_cmc joint).
        PIP joints are auto-mimicked via the 4-bar linkage formula.

    Observation (57-dim, hand-frame relative -- arm state NOT included):
        - hand joint positions (6)
        - hand joint velocities (6)
        - FSR touch sensors (30)
        - object position in palm frame (3)
        - object orientation in palm frame (6) [first two cols of rotmat]
        - object linear velocity in palm frame (3)
        - object angular velocity in palm frame (3)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Actuator indices in the UR10e+Hand model:
    # 0-5: arm (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
    # 6: index_mcp  7: index_pip  8: middle_mcp  9: middle_pip
    # 10: ring_mcp  11: ring_pip  12: pinky_mcp  13: pinky_pip
    # 14: thumb_flexor (thumb_mcp joint)  15: thumb_rotator (thumb_cmc joint)
    _ARM_CTRL_IDS = [0, 1, 2, 3, 4, 5]
    # Frozen arm pose, matches the "grasp_ready" keyframe in the scene XML.
    _ARM_HOLD_CTRL = np.array([-1.6, -1.6, 1.7, -1.5, -1.5708, 0.0])

    # Hand actuators the policy commands, in action-vector order:
    # [idx_mcp, mid_mcp, ring_mcp, pinky_mcp, thumb_flexor, thumb_rotator]
    _CTRL_IDS = [6, 8, 10, 12, 14, 15]
    # PIP mimic source (MCP actuators) -> target (PIP actuators)
    _MCP_CTRL_IDS = [6, 8, 10, 12]
    _PIP_CTRL_IDS = [7, 9, 11, 13]

    # qpos addresses for the 6 controlled hand DOFs, in the same order as
    # _CTRL_IDS. In the UR10e+Hand model the hand joints occupy qpos 6..15:
    # idx_mcp=6, idx_pip=7, mid_mcp=8, mid_pip=9, ring_mcp=10, ring_pip=11,
    # pinky_mcp=12, pinky_pip=13, thumb_cmc=14, thumb_mcp=15.
    # Note: actuator 14 (thumb_flexor) drives joint thumb_mcp (qpos 15);
    # actuator 15 (thumb_rotator) drives joint thumb_cmc (qpos 14).
    _HAND_QPOS_IDS = [6, 8, 10, 12, 15, 14]

    # 4-bar linkage mimic: pip_ctrl = mcp_ctrl * slope + intercept
    _MIMIC_SLOPE = 1.05851325
    _MIMIC_INTERCEPT = 0.72349796

    # FSR per-finger groupings (6 sensors each, sensordata[0:30])
    _FINGER_FSR_RANGES = [
        (0, 6),    # index
        (6, 12),   # middle
        (12, 18),  # ring
        (18, 24),  # pinky
        (24, 30),  # thumb
    ]

    def __init__(
        self,
        scene_path: Optional[str] = None,
        frame_skip: int = 10,
        max_episode_steps: int = 300,
        render_mode: Optional[str] = None,
        obj_pos_range: Tuple[float, float] = (0.0, 0.0),
        reward_weights: Optional[Dict[str, float]] = None,
        sigma_d: float = 0.07,
        sigma_f: float = 5.0,
        lift_threshold: float = 0.03,
        hold_steps: int = 25,
    ):
        super().__init__()

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.obj_pos_range = obj_pos_range
        self.sigma_d = sigma_d
        self.sigma_f = sigma_f
        self.lift_threshold = lift_threshold
        self.hold_steps = hold_steps

        self.rw = {
            "approach": 0.2,
            "contact": 0.5,
            "lift": 0.5,
            "success": 5.0,
            "smooth": 0.001,
        }
        if reward_weights:
            self.rw.update(reward_weights)

        scene = scene_path or os.path.normpath(_DEFAULT_SCENE)
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)

        self._palm_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ability_hand"
        )
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object"
        )
        self._obj_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint"
        )
        self._obj_qpos_adr = int(self.model.jnt_qposadr[self._obj_joint_id])
        self._obj_qvel_adr = int(self.model.jnt_dofadr[self._obj_joint_id])

        # Look up keyframe by name (the included ur10e.xml has its own
        # "home" keyframe at index 0, so we must select "grasp_ready").
        self._key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "grasp_ready"
        )
        if self._key_id < 0:
            raise RuntimeError("Scene XML is missing the 'grasp_ready' keyframe")

        self._palm_pos_adr = int(
            self.model.sensor_adr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_pos")
            ]
        )
        # palm_xaxis/yaxis/zaxis sensors are contiguous after palm_pos
        self._palm_xaxis_adr = int(
            self.model.sensor_adr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_xaxis")
            ]
        )

        # Object spawns FLOATING at z=0.500 (the finger-thumb closure point).
        # Gravity makes it fall unless gripped. "Lift" reward = held near spawn;
        # success = sustained holding within `lift_threshold` of spawn z.
        self._obj_spawn_z = 0.530
        self._obj_rest_z = self._obj_spawn_z   # legacy alias (unused, kept for compat)
        self._fallen_z = 0.30  # object hit floor_table -> terminate
        self._default_obj_pos = np.array([-0.155, 0.620, self._obj_spawn_z])

        # qvel addresses for the 6 controlled hand DOFs (1 dof per hinge joint)
        self._hand_qvel_ids = [
            int(self.model.jnt_dofadr[self.model.jnt_qposadr.tolist().index(q)])
            for q in self._HAND_QPOS_IDS
        ]
        # Simpler/equivalent: dofadr for hinge joint = qposadr - 1 because the
        # only freejoint (object) comes AFTER all hand joints in this model,
        # so the arm+hand hinge joints have dofadr == qposadr. We trust the
        # jnt_dofadr lookup above instead.

        # Control limits for the 6 hand actuators
        ctrl_low = self.model.actuator_ctrlrange[self._CTRL_IDS, 0]
        ctrl_high = self.model.actuator_ctrlrange[self._CTRL_IDS, 1]
        self._ctrl_low = ctrl_low
        self._ctrl_high = ctrl_high
        self._ctrl_center = (ctrl_high + ctrl_low) / 2
        self._ctrl_range = (ctrl_high - ctrl_low) / 2

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # 6 + 6 + 30 + 3 + 6 + 3 + 3 = 57
        obs_dim = 6 + 6 + 30 + 3 + 6 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        self._viewer = None
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self._step_count = 0
        self._hold_counter = 0
        self._prev_action = np.zeros(6, dtype=np.float32)

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

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_id)

        # Optional object xy randomization around the default grasp-zone center
        dx = self.np_random.uniform(-self.obj_pos_range[0], self.obj_pos_range[0])
        dy = self.np_random.uniform(-self.obj_pos_range[1], self.obj_pos_range[1])
        obj_pos = self._default_obj_pos + np.array([dx, dy, 0.0])
        self.data.qpos[self._obj_qpos_adr : self._obj_qpos_adr + 3] = obj_pos
        self.data.qpos[self._obj_qpos_adr + 3 : self._obj_qpos_adr + 7] = [
            1.0, 0.0, 0.0, 0.0
        ]
        self.data.qvel[self._obj_qvel_adr : self._obj_qvel_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._hold_counter = 0
        self._prev_action = np.zeros(6, dtype=np.float32)

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Hold the arm at the grasp-ready pose (stiff position control)
        for i, ctrl_id in enumerate(self._ARM_CTRL_IDS):
            self.data.ctrl[ctrl_id] = self._ARM_HOLD_CTRL[i]

        # Map policy action to hand ctrl
        ctrl_targets = self._ctrl_center + action * self._ctrl_range
        for i, ctrl_id in enumerate(self._CTRL_IDS):
            self.data.ctrl[ctrl_id] = ctrl_targets[i]

        # PIP mimic
        for mcp_id, pip_id in zip(self._MCP_CTRL_IDS, self._PIP_CTRL_IDS):
            self.data.ctrl[pip_id] = (
                self.data.ctrl[mcp_id] * self._MIMIC_SLOPE + self._MIMIC_INTERCEPT
            )

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated = self._check_terminated(reward_info)
        truncated = self._step_count >= self.max_episode_steps

        info = self._get_info()
        info.update(reward_info)
        self._prev_action = action
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
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
    # Helpers
    # ------------------------------------------------------------------

    def _palm_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (palm_pos in world, R_world_from_palm 3x3 rotation)."""
        palm_pos = self.data.sensordata[
            self._palm_pos_adr : self._palm_pos_adr + 3
        ].copy()
        # Three framex/y/zaxis sensors each give the world-frame components
        # of the palm body's local x, y, z axes -> columns of R_world_from_palm.
        axes = self.data.sensordata[
            self._palm_xaxis_adr : self._palm_xaxis_adr + 9
        ].reshape(3, 3).T
        return palm_pos, axes

    def _get_obs(self) -> np.ndarray:
        hand_pos = self.data.qpos[self._HAND_QPOS_IDS].copy()
        hand_vel = self.data.qvel[self._hand_qvel_ids].copy()

        fsr = self.data.sensordata[0:30].copy()

        obj_pos_w = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ].copy()
        obj_quat = self.data.qpos[
            self._obj_qpos_adr + 3 : self._obj_qpos_adr + 7
        ].copy()
        obj_vel_w = self.data.qvel[
            self._obj_qvel_adr : self._obj_qvel_adr + 6
        ].copy()

        palm_pos_w, R_wp = self._palm_pose()
        R_pw = R_wp.T

        rel_pos = R_pw @ (obj_pos_w - palm_pos_w)

        R_obj = np.zeros(9)
        mujoco.mju_quat2Mat(R_obj, obj_quat)
        R_obj = R_obj.reshape(3, 3)
        R_rel = R_pw @ R_obj
        obj_rot6 = R_rel[:, :2].reshape(-1)

        obj_linvel_p = R_pw @ obj_vel_w[0:3]
        obj_angvel_p = R_pw @ obj_vel_w[3:6]

        return np.concatenate([
            hand_pos, hand_vel, fsr,
            rel_pos, obj_rot6, obj_linvel_p, obj_angvel_p,
        ])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self, action: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        palm_pos_w, _ = self._palm_pose()
        obj_pos = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ]

        distance = float(np.linalg.norm(palm_pos_w - obj_pos))
        r_approach = float(np.exp(-distance / self.sigma_d))

        fsr = self.data.sensordata[0:30]
        per_finger = np.array([
            np.tanh(np.sum(fsr[lo:hi]) / self.sigma_f)
            for lo, hi in self._FINGER_FSR_RANGES
        ])
        r_contact_raw = float(per_finger.mean())
        r_contact = r_approach * r_contact_raw  # gated by approach

        obj_z = float(obj_pos[2])
        # "Lift" = the object is being held near its spawn z (i.e. the agent
        # is supporting it against gravity). r_lift is a tent function around
        # the spawn z: peaks at 1.0 when obj_z == spawn_z, falls off as it
        # drops below.
        drop = max(0.0, self._obj_spawn_z - obj_z)
        r_lift = float(np.clip(1.0 - drop / 0.05, 0.0, 1.0))  # zero once it's fallen 5 cm
        # For info compatibility, also expose the standard "lift_height"
        # (positive only if the agent has raised the object above spawn).
        lift_height = max(0.0, obj_z - self._obj_spawn_z)

        # Sustained-hold success: object stays within lift_threshold of spawn z
        if abs(obj_z - self._obj_spawn_z) < self.lift_threshold:
            self._hold_counter += 1
        else:
            self._hold_counter = 0
        success_fired = self._hold_counter >= self.hold_steps
        r_success = 1.0 if success_fired else 0.0

        d_action = action - self._prev_action
        r_smooth = float(np.sum(d_action ** 2))

        reward = (
            self.rw["approach"] * r_approach
            + self.rw["contact"]  * r_contact
            + self.rw["lift"]     * r_lift
            + self.rw["success"]  * r_success
            - self.rw["smooth"]   * r_smooth
        )

        info = {
            "r_approach": r_approach,
            "r_contact": r_contact,
            "r_contact_raw": r_contact_raw,
            "r_lift": r_lift,
            "r_success": r_success,
            "r_smooth_pen": -r_smooth,
            "distance": distance,
            "fsr_total": float(np.sum(fsr)),
            "lift_height": lift_height,
            "hold_counter": self._hold_counter,
            "success_fired": bool(success_fired),
        }
        return float(reward), info

    # ------------------------------------------------------------------
    # Termination & info
    # ------------------------------------------------------------------

    def _check_terminated(self, reward_info: Dict[str, float]) -> bool:
        if reward_info.get("success_fired", False):
            return True
        obj_pos = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ]
        # Object fell off the shelf to the floor.
        if obj_pos[2] < self._fallen_z:
            return True
        # Object knocked far from the workspace center.
        if np.linalg.norm(obj_pos[:2] - self._default_obj_pos[:2]) > 0.4:
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        palm_pos_w, _ = self._palm_pose()
        obj_pos = self.data.qpos[
            self._obj_qpos_adr : self._obj_qpos_adr + 3
        ]
        return {
            "step": self._step_count,
            "palm_pos": palm_pos_w.copy(),
            "obj_pos": obj_pos.copy(),
            "obj_z": float(obj_pos[2]),
        }
