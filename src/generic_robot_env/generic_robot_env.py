from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import mujoco
import numpy as np
from gym_hil.controllers import opspace
from gym_hil.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from gymnasium import spaces


@dataclass
class RobotConfig:
    """Configuration for a generic robot environment."""

    robot_name: str
    xml_path: Path
    joint_names: list[str]
    actuator_names: list[str]
    ee_site_name: str
    gripper_actuator_name: str | None = None
    home_position: np.ndarray | None = None
    cartesian_bounds: np.ndarray | None = None
    camera_names: list[str] = field(default_factory=list[str])

    # Auto-detected later if None
    dof_ids: np.ndarray | None = None
    actuator_ids: np.ndarray | None = None
    ee_site_id: int | None = None
    gripper_actuator_id: int | None = None

    @staticmethod
    def from_xml(xml_path: Path, robot_name: str) -> RobotConfig:
        """Auto-configure from an XML file."""
        return extract_config_from_xml(xml_path, robot_name)


def extract_config_from_xml(xml_path: Path, robot_name: str) -> RobotConfig:
    """load the model and auto-detect joints, actuators, and 'home' keyframe."""
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())

    # Auto-detect joints
    # Strategy: Find joints associated with the robot body.
    # For now, we'll take all non-free joints or try to filter by name if possible.
    # A better heuristic for "manipulator joints" might be needed.
    # Here we assume all 1-DOF joints are part of the robot arm unless specified.

    joint_names: list[str] = []
    for i in range(model.njnt):
        jnt_type = model.jnt_type[i]
        if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                joint_names.append(name)

    # Auto-detect actuators
    actuator_names: list[str] = []
    gripper_actuator_name = None

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            # Simple heuristic for gripper: verify if "gripper" or "finger" in name
            if "gripper" in name or "finger" in name:
                gripper_actuator_name = name
            else:
                actuator_names.append(name)

    # Auto-detect end-effector site
    # Look for a site named "end_effector", "ee", "tool_center", "attachment_site", etc.
    ee_candidates = [
        "attachment_site",
        "pinch",
        "ee",
        "end_effector",
        "tool_center",
        "tcp",
    ]
    ee_site_name = None
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if name in ee_candidates:
            ee_site_name = name
            break

    # If not found, use the last site? Or fail? Let's default to the last site on the
    # last body if unsure, but for now let's raise/warn if not found or pick the
    # first distinct candidate.
    if ee_site_name is None and model.nsite > 0:
        # Fallback: check for substring match
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name and any(c in name for c in ee_candidates):
                ee_site_name = name
                break

    if ee_site_name is None and model.nsite > 0:
        ee_site_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE, model.nsite - 1
        )

    # Auto-detect home position from keyframe
    home_position = None
    if model.nkey > 0:
        # Look for a keyframe named "home"
        for i in range(model.nkey):
            # Keyframe names are not directly exposed in python bindings
            # easily as strings in older versions, but let's check if we
            # can access it.
            # mujoco 3.x bindings: model.key(i).name is not standard.
            # key_name is missing.
            # However, we can use id2name with mjOBJ_KEY
            key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name == "home":
                # Get the qpos for this keyframe
                # model.key_qpos is (nkey, nq)
                # We need to filter for the relevant joints.
                # This is tricky because key_qpos includes ALL joints
                # (including free joints, etc)
                # We will extract it in the env init where we have the
                # joint ids. For now, store the raw key_qpos or just the fact we
                # have a home key.  But here we want the numpy array of shape
                # (n_joints,).
                pass

    # Cameras
    camera_names: list[str] = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            camera_names.append(name)

    return RobotConfig(
        robot_name=robot_name,
        xml_path=xml_path,
        joint_names=joint_names,
        actuator_names=actuator_names,
        ee_site_name=ee_site_name if ee_site_name else "ee",  # Default fallback
        gripper_actuator_name=gripper_actuator_name,
        home_position=home_position,  # Will be handled in env if None, or updated later
        camera_names=camera_names,
    )


class GenericRobotEnv(MujocoGymEnv[dict[str, Any], np.ndarray]):
    """Generic robot environment supporting OSC and joint control."""

    _rob_joint_ids: np.ndarray
    _rob_actuator_ids: np.ndarray
    _rob_ee_site_id: int
    _rob_qpos_indices: np.ndarray
    _rob_dof_indices: np.ndarray
    _rob_gripper_actuator_id: int | None
    _rob_home_position: np.ndarray
    _target_pos: np.ndarray
    _target_quat: np.ndarray
    _cartesian_bounds: np.ndarray
    _render_specs: GymRenderingSpec
    camera_ids: list[int]
    _viewer: mujoco.Renderer

    def __init__(
        self,
        robot_config: RobotConfig,
        control_mode: Literal["osc", "joint"] = "osc",
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec | None = None,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        camera_id: int | None = None,  # Override default camera
    ):
        if render_spec is None:
            render_spec = GymRenderingSpec()

        super().__init__(
            xml_path=robot_config.xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self.robot_config = robot_config
        self.control_mode = control_mode
        self.image_obs = image_obs
        self.render_mode = render_mode

        # --- Resolve IDs ---
        # Rename internal variables to avoid clashes with base classes
        self._rob_joint_ids = np.array(
            [self._model.joint(name).id for name in robot_config.joint_names],
            dtype=np.int32,
        )
        self._rob_actuator_ids = np.array(
            [self._model.actuator(name).id for name in robot_config.actuator_names],
            dtype=np.int32,
        )

        # Site resolution with fallback
        try:
            self._rob_ee_site_id = self._model.site(robot_config.ee_site_name).id
        except Exception:
            # Fallback: find any site or use body frame if needed
            if self._model.nsite > 0:
                self._rob_ee_site_id = 0
            else:
                # Use the last body as the site frame? Or just error?
                self._rob_ee_site_id = (
                    self._model.nbody - 1
                )  # Simple fallback to last body center

        # Resolve qpos and dof indices for data access
        # qpos indices (for position)
        self._rob_qpos_indices = np.array(
            [self._model.jnt_qposadr[i] for i in self._rob_joint_ids], dtype=np.int32
        )
        # dof indices (for velocity and torque)
        self._rob_dof_indices = np.array(
            [self._model.jnt_dofadr[i] for i in self._rob_joint_ids], dtype=np.int32
        )

        self._rob_gripper_actuator_id = None
        if robot_config.gripper_actuator_name:
            self._rob_gripper_actuator_id = self._model.actuator(
                robot_config.gripper_actuator_name
            ).id

        # --- Resolve Home Position ---
        if robot_config.home_position is None:
            # Try to load from "home" keyframe if available and not set
            key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0:
                # Extract the qpos values corresponding to our joints
                # model.key_qpos is (nkey, nq)
                full_qpos = self._model.key_qpos[key_id]

                # We need to map joint ids to qpos indices.
                # For hinge/slide joints, qpos_adr gives the index in qpos.
                self._rob_home_position = full_qpos[self._rob_qpos_indices]
            else:
                # Default to zeros if no home keyframe and no config provided
                self._rob_home_position = np.zeros(len(robot_config.joint_names))
        else:
            self._rob_home_position = robot_config.home_position

        # --- Bounds ---
        if robot_config.cartesian_bounds is None:
            # Default loose bounds
            self._cartesian_bounds = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]])
        else:
            self._cartesian_bounds = robot_config.cartesian_bounds

        # --- Cameras ---
        # If camera_names are provided in config, map them to IDs
        self.camera_ids = []
        if robot_config.camera_names:
            for cam_name in robot_config.camera_names:
                self.camera_ids.append(
                    mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                )

        # Determine main render camera
        if camera_id is not None:
            self._render_specs = GymRenderingSpec(
                height=render_spec.height,
                width=render_spec.width,
                camera_id=camera_id,
                mode=render_spec.mode,
            )
        elif self.camera_ids:
            # Use the first found camera as default for generic rendering
            self._render_specs = GymRenderingSpec(
                height=render_spec.height,
                width=render_spec.width,
                camera_id=self.camera_ids[0],
                mode=render_spec.mode,
            )

        # Initialize renderer
        self._viewer = mujoco.Renderer(
            self.model, height=render_spec.height, width=render_spec.width
        )

        # Setup spaces
        self._setup_observation_space()
        self._setup_action_space()

    def _setup_observation_space(self):
        """Setup observation space."""
        # Generic observation: Joint positions, velocities, EE pose
        # If gripper exists, add gripper pose/state

        n_joints = len(self._rob_joint_ids)

        # Similar structure to FrankaGymEnv for compatibility
        agent_pos_space = {
            "joint_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_joints,), dtype=np.float32
            ),
            "joint_vel": spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_joints,), dtype=np.float32
            ),
            "tcp_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),  # Pos + Quat (ee_pose)
            "tcp_vel": spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            ),  # Linear + Angular vel
        }

        if self._rob_gripper_actuator_id is not None:
            agent_pos_space["gripper_pose"] = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        base_obs_space: dict[str, spaces.Space[Any]] = {
            "agent_pos": spaces.Dict(cast(dict[str, Any], agent_pos_space))
        }

        if self.image_obs and self.camera_ids:
            # Add images
            pixels_space: dict[str, spaces.Space[Any]] = {}
            for cam_name in self.robot_config.camera_names:
                pixels_space[cam_name] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._render_specs.height, self._render_specs.width, 3),
                    dtype=np.uint8,
                )
            base_obs_space["pixels"] = spaces.Dict(cast(dict[str, Any], pixels_space))

        self.observation_space = spaces.Dict(cast(dict[str, Any], base_obs_space))

    def _setup_action_space(self):
        """Setup action space based on control mode."""
        if self.control_mode == "osc":
            # OSC: x, y, z, rx, ry, rz (delta) + gripper
            # Assuming 6D control for EE + 1D for gripper (if exists)

            low = np.array([-1.0] * 6, dtype=np.float32)
            high = np.array([1.0] * 6, dtype=np.float32)

            if self._rob_gripper_actuator_id is not None:
                low = np.append(low, -1.0)
                high = np.append(high, 1.0)

            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        elif self.control_mode == "joint":
            # Joint position/velocity delta
            n_actuators = len(self._rob_actuator_ids)
            if self._rob_gripper_actuator_id is not None:
                n_actuators += 1

            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        # Reset robot to home configuration
        self._data.qpos[self._rob_qpos_indices] = self._rob_home_position
        self._data.qvel[self._rob_dof_indices] = 0.0
        self._data.ctrl[:] = 0.0  # Reset all controls

        mujoco.mj_forward(self._model, self._data)

        # If OSC, reset mocap/target to current EE pose
        if self.control_mode == "osc":
            # Initialize target from current state regardless of mocap existence
            # This ensures consistent starting state for controller
            ee_pos = self._data.site_xpos[self._rob_ee_site_id].copy()
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, self._data.site_xmat[self._rob_ee_site_id])

            self._target_pos = ee_pos.copy()
            self._target_quat = ee_quat.copy()

            if self._model.nmocap > 0:
                self._data.mocap_pos[0] = ee_pos
                self._data.mocap_quat[0] = ee_quat

        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action_space = cast(Any, self.action_space)
        action = np.clip(action, action_space.low, action_space.high)

        if self.control_mode == "osc":
            self._step_osc(action)
        elif self.control_mode == "joint":
            self._step_joint(action)

        # Observations
        obs = self._get_obs()
        reward = 0.0  # Placeholder
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _step_osc(self, action: np.ndarray) -> None:
        # Action: [x, y, z, rx, ry, rz, gripper]
        delta_pos = action[:3] * 0.05  # Scale delta
        # TODO: handle rotation delta properly

        gripper_action = 0.0
        if self._rob_gripper_actuator_id is not None:
            gripper_action = action[-1]

        # Update target
        if self._model.nmocap == 0:
            pass

        if self._model.nmocap > 0:
            # Use mocap
            self._data.mocap_pos[0] += delta_pos
            # Clamp to bounds
            self._data.mocap_pos[0] = np.clip(
                self._data.mocap_pos[0],
                self._cartesian_bounds[0],
                self._cartesian_bounds[1],
            )
            target_pos = self._data.mocap_pos[0]
            target_quat = self._data.mocap_quat[
                0
            ]  # Assume Orientation fixed or update it if needed
        else:
            # Use internal state
            self._target_pos += delta_pos
            self._target_pos = np.clip(
                self._target_pos, self._cartesian_bounds[0], self._cartesian_bounds[1]
            )
            target_pos = self._target_pos
            target_quat = self._target_quat

        # Apply gripper
        if self._rob_gripper_actuator_id is not None:
            # Map -1..1 to 0..255 or similar depends on actuator config
            # Franka env uses 0..255 and scales it.
            # We'll assume a simpler -1..1 -> ctrl_range or similar.
            # Actually FrankaGymEnv logic:
            # g = self._data.ctrl[self._gripper_ctrl_id] / MAX_GRIPPER_COMMAND
            # ng = np.clip(g + grasp_command, 0.0, 1.0)
            # self._data.ctrl[self._gripper_ctrl_id] = ng * MAX_GRIPPER_COMMAND
            # We should probably respect the actuator control range in the model.
            ctrl_range = self._model.actuator_ctrlrange[self._rob_gripper_actuator_id]
            # If range is defined (not 0,0)
            if ctrl_range[1] > ctrl_range[0]:
                # Map action -1..1 to range
                val = (gripper_action + 1) / 2 * (
                    ctrl_range[1] - ctrl_range[0]
                ) + ctrl_range[0]
                self._data.ctrl[self._rob_gripper_actuator_id] = val
            else:
                # Just pass it through or normalized
                self._data.ctrl[self._rob_gripper_actuator_id] = gripper_action

        # Control loop
        for _ in range(self._n_substeps):
            # Nullspace target assumes home_position aligns with qpos of
            # controlled joints
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._rob_ee_site_id,
                dof_ids=self._rob_dof_indices,
                pos=target_pos,
                ori=target_quat,
                joint=self._rob_home_position,
                gravity_comp=True,
            )
            self._data.ctrl[self._rob_actuator_ids] = tau
            mujoco.mj_step(self._model, self._data)

    def _step_joint(self, action: np.ndarray) -> None:
        # Joint pd or velocity control
        # This is a placeholder for joint control.
        # For now we will just apply action as torque or position delta?
        # Typically "joint" means setting qpos targets or ctrl directly.

        # If the actuators are position servos, we add delta to current qpos.
        # If they are torque motors, we add torque.
        # For simplicity in Generic, let's assume direct control (action = ctrl).

        n_act = len(self._rob_actuator_ids)
        self._data.ctrl[self._rob_actuator_ids] = action[:n_act]  # simplistic

        if self._rob_gripper_actuator_id is not None and len(action) > n_act:
            self._data.ctrl[self._rob_gripper_actuator_id] = action[n_act]

        mujoco.mj_step(self._model, self._data)
        # We need to sub-step if physics_dt != control_dt?
        # FrankaGymEnv does manual substep loop only for OSC because OSC
        # computes tau every step.
        # For pure step, mj_step simulates one timestep defined in model.opt.timestep.
        # But we want to simulate control_dt.

        # If we use one mj_step, we advance by physics_dt.
        # We need to loop.
        for _ in range(self._n_substeps - 1):
            mujoco.mj_step(self._model, self._data)

    def _get_obs(self) -> dict[str, Any]:
        # Read sensors if available, or qpos/qvel
        qpos = self._data.qpos[self._rob_qpos_indices].astype(np.float32)
        qvel = self._data.qvel[self._rob_dof_indices].astype(np.float32)

        # TCP Pose
        ee_pos = self._data.site_xpos[self._rob_ee_site_id].flatten().astype(np.float32)
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, self._data.site_xmat[self._rob_ee_site_id])
        ee_pose = np.concatenate([ee_pos, ee_quat]).astype(np.float32)

        # TCP Vel (linear + angular)
        # site_xvelp and site_xvelr are not directly exposed in older mujoco
        # bindings easily as separate arrays?
        # Actually in mjData.site_xvelp yes.
        # But we need to check if they are computed. Usually yes if
        # MjModel.opt.enableflags is set or forward called.
        # Alternatively compute Jacobian J

        # Using built-in site velocity if available
        # Note: site_xvelp/r needs mj_kinematics + mj_comPos + mj_jac or similar.
        # But after mj_forward they should be valid if we enable them?
        # Let's compute via Jacobian to be safe and consistent with opspace code
        # if needed, or rely on data.
        # Actually, let's use the object velocity if simple enough.

        # For robustness let's just use what's available or zeros if tricky.
        # But FrankaEnv had `tcp_vel`.
        # FrankaEnv implementation commented it out:
        # "tcp_vel = self._data.sensor("2f85/pinch_vel").data"
        # So maybe they rely on sensors. Generic might not have sensors.

        # Let's compute it: J*qvel
        jac_p = np.zeros((3, self._model.nv))
        jac_r = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jac_p, jac_r, self._rob_ee_site_id)

        dq = self._data.qvel
        # We need full qvel vector, not just dof_ids if they are subset?
        # If dof_ids covers all moving joints then fine.
        # But if free joints exist they matter.
        # However for manipulators usually base is fixed.

        ee_vel_lin = jac_p @ dq
        ee_vel_ang = jac_r @ dq
        ee_vel = np.concatenate([ee_vel_lin, ee_vel_ang]).astype(np.float32)

        agent_pos = {
            "joint_pos": qpos,
            "joint_vel": qvel,
            "tcp_pose": ee_pose,
            "tcp_vel": ee_vel,
        }

        if self._rob_gripper_actuator_id is not None:
            # Gripper state. For now just control value or simple 0-1
            # If we can map to width it would be better.
            # Using control input as proxy if sensor not available.
            agent_pos["gripper_pose"] = np.array(
                [self._data.ctrl[self._rob_gripper_actuator_id]], dtype=np.float32
            )

        obs = {"agent_pos": agent_pos}

        if self.image_obs and self.camera_ids:
            pixels = {}
            for i, cam_id in enumerate(self.camera_ids):
                self._viewer.update_scene(self._data, camera=cam_id)
                pixels[self.robot_config.camera_names[i]] = self._viewer.render()
            obs["pixels"] = pixels

        return obs
