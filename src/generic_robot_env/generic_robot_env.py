from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import numpy as np
from gym_hil.controllers import opspace
from gym_hil.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from gymnasium import spaces

MAX_GRIPPER_COMMAND = 255.0

if TYPE_CHECKING:
    MujocoGymEnvBase = MujocoGymEnv[dict[str, Any], np.ndarray]
else:
    MujocoGymEnvBase = MujocoGymEnv


@dataclass
class RobotConfig:
    """Configuration for a generic robot environment."""

    robot_name: str
    xml_path: Path
    joint_names: list[str]
    actuator_names: list[str]
    end_effector_site_name: str
    gripper_actuator_name: str | None = None
    home_position: np.ndarray | None = None
    cartesian_bounds: np.ndarray | None = None
    camera_names: list[str] = field(default_factory=list[str])

    # Auto-detected later if None
    dof_ids: np.ndarray | None = None
    actuator_ids: np.ndarray | None = None
    end_effector_site_id: int | None = None
    gripper_actuator_id: int | None = None

    @staticmethod
    def from_xml(xml_path: Path, robot_name: str) -> RobotConfig:
        """Auto-configure from an XML file."""
        return extract_config_from_xml(xml_path, robot_name)


@dataclass(frozen=True)
class EndEffectorRef:
    """Resolved end-effector reference in the MuJoCo model."""

    obj_type: Literal["site", "body"]
    obj_id: int
    name: str


def extract_config_from_xml(xml_path: Path, robot_name: str) -> RobotConfig:
    """Load the model and auto-detect joints, actuators, and 'home' keyframe."""
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

    # Detect actuators and identify gripper actuators based on naming heuristics.
    actuator_names: list[str] = []
    gripper_actuator_name = None

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            if "gripper" in name or "finger" in name:
                gripper_actuator_name = name
            else:
                actuator_names.append(name)

    # Detect end-effector site using a list of common candidate names.
    end_effector_candidates = [
        "attachment_site",
        "pinch",
        "ee",
        "end_effector",
        "tool_center",
        "tcp",
    ]
    end_effector_site_name = None
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if name in end_effector_candidates:
            end_effector_site_name = name
            break

    # If no exact match is found, fallback to substring matches.
    if end_effector_site_name is None and model.nsite > 0:
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name and any(c in name for c in end_effector_candidates):
                end_effector_site_name = name
                break

    # Final fallback: use the last site index.
    if end_effector_site_name is None and model.nsite > 0:
        end_effector_site_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE, model.nsite - 1
        )

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
        end_effector_site_name=end_effector_site_name
        if end_effector_site_name
        else "end_effector",
        gripper_actuator_name=gripper_actuator_name,
        camera_names=camera_names,
    )


class GenericRobotEnv(MujocoGymEnvBase):
    """Generic robot base environment with robot-control API.

    This class focuses on reusable robot mechanics and controller integration,
    including action application, robot state extraction, reset helpers, and
    rendering. Task-specific reward/termination logic should be implemented in
    subclasses.
    """

    _joint_ids: np.ndarray
    _actuator_ids: np.ndarray
    _end_effector: EndEffectorRef
    _joint_qpos_indices: np.ndarray
    _dof_ids: np.ndarray
    _gripper_actuator_id: int | None
    _home_position: np.ndarray
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
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / control_dt)),
        }

        # Resolve joint and actuator IDs
        self._joint_ids = np.array(
            [self._model.joint(name).id for name in robot_config.joint_names],
            dtype=np.int32,
        )
        self._actuator_ids = np.array(
            [self._model.actuator(name).id for name in robot_config.actuator_names],
            dtype=np.int32,
        )

        self._end_effector = self._resolve_end_effector_ref(
            robot_config.end_effector_site_name
        )
        if self.control_mode == "osc" and self._end_effector.obj_type != "site":
            self.control_mode = "joint"

        # Map joint IDs to qpos and DOF indices
        self._joint_qpos_indices = np.array(
            [self._model.jnt_qposadr[i] for i in self._joint_ids], dtype=np.int32
        )
        self._dof_ids = np.array(
            [self._model.jnt_dofadr[i] for i in self._joint_ids], dtype=np.int32
        )

        self._gripper_actuator_id = None
        if robot_config.gripper_actuator_name:
            self._gripper_actuator_id = self._model.actuator(
                robot_config.gripper_actuator_name
            ).id

        # Resolve home position from configuration or keyframe
        if robot_config.home_position is None:
            key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0:
                full_qpos = self._model.key_qpos[key_id]
                self._home_position = full_qpos[self._joint_qpos_indices]
            else:
                self._home_position = np.zeros(len(robot_config.joint_names))
        else:
            self._home_position = robot_config.home_position

        # Bounds for Cartesian control
        if robot_config.cartesian_bounds is None:
            self._cartesian_bounds = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]])
        else:
            self._cartesian_bounds = robot_config.cartesian_bounds

        # Map camera names to IDs
        self.camera_ids = []
        if robot_config.camera_names:
            for cam_name in robot_config.camera_names:
                self.camera_ids.append(
                    mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                )

        # Specify rendering camera based on user choice or default
        if camera_id is not None:
            self._render_specs = GymRenderingSpec(
                height=render_spec.height,
                width=render_spec.width,
                camera_id=camera_id,
                mode=render_spec.mode,
            )
        elif self.camera_ids:
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
        """Setup robot observation space and optional camera image observations."""
        joint_dim = self._joint_qpos_indices.shape[0]
        agent_space = spaces.Dict(
            {
                "joint_pos": spaces.Box(
                    -np.inf, np.inf, (joint_dim,), dtype=np.float32
                ),
                "joint_vel": spaces.Box(
                    -np.inf, np.inf, (joint_dim,), dtype=np.float32
                ),
                "tcp_pose": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
                "tcp_vel": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                "gripper_pose": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )
        base_obs_space: dict[str, spaces.Space[Any]] = {"agent_pos": agent_space}

        if self.image_obs and self.camera_ids:
            # If image observations are enabled, include RGB data from specified cameras
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
        """Setup action space for operational-space or joint control."""
        if self.control_mode == "osc":
            # Action consists of [x, y, z, rx, ry, rz] deltas and optional
            # gripper command.
            low = np.array([-1.0] * 6, dtype=np.float32)
            high = np.array([1.0] * 6, dtype=np.float32)

            if self._gripper_actuator_id is not None:
                low = np.append(low, -1.0)
                high = np.append(high, 1.0)

            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        elif self.control_mode == "joint":
            n_actuators = len(self._actuator_ids)
            if self._gripper_actuator_id is not None:
                n_actuators += 1

            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32
            )

    def _resolve_end_effector_ref(self, preferred_name: str) -> EndEffectorRef:
        body_obj_type = 1
        site_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_SITE,
            preferred_name,
        )
        if site_id >= 0:
            site_name = mujoco.mj_id2name(
                self._model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_id,
            )
            return EndEffectorRef(
                obj_type="site",
                obj_id=site_id,
                name=site_name if site_name else preferred_name,
            )

        if self._model.nsite > 0:
            fallback_site_name = mujoco.mj_id2name(
                self._model,
                mujoco.mjtObj.mjOBJ_SITE,
                0,
            )
            return EndEffectorRef(
                obj_type="site",
                obj_id=0,
                name=fallback_site_name if fallback_site_name else "site_0",
            )

        body_candidates = [
            preferred_name,
            "attachment_site",
            "pinch",
            "ee",
            "end_effector",
            "tool_center",
            "tcp",
            "gripper",
            "hand",
            "flange",
        ]
        for candidate in body_candidates:
            body_id = mujoco.mj_name2id(self._model, body_obj_type, candidate)
            if body_id >= 0:
                body_name = mujoco.mj_id2name(self._model, body_obj_type, body_id)
                return EndEffectorRef(
                    obj_type="body",
                    obj_id=body_id,
                    name=body_name if body_name else candidate,
                )

        fallback_body_id = self._model.nbody - 1
        fallback_body_name = mujoco.mj_id2name(
            self._model,
            body_obj_type,
            fallback_body_id,
        )
        return EndEffectorRef(
            obj_type="body",
            obj_id=fallback_body_id,
            name=(
                fallback_body_name if fallback_body_name else f"body_{fallback_body_id}"
            ),
        )

    def _get_end_effector_position(self) -> np.ndarray:
        if self._end_effector.obj_type == "site":
            return self._data.site_xpos[self._end_effector.obj_id]
        return cast(Any, self._data).xpos[self._end_effector.obj_id]

    def _get_end_effector_rotation_matrix(self) -> np.ndarray:
        if self._end_effector.obj_type == "site":
            return self._data.site_xmat[self._end_effector.obj_id]
        return cast(Any, self._data).xmat[self._end_effector.obj_id]

    def _get_end_effector_quaternion(self) -> np.ndarray:
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, self._get_end_effector_rotation_matrix())
        return quaternion

    def _get_end_effector_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        jacobian_position = np.zeros((3, self._model.nv))
        jacobian_rotation = np.zeros((3, self._model.nv))
        if self._end_effector.obj_type == "site":
            mujoco.mj_jacSite(
                self._model,
                self._data,
                jacobian_position,
                jacobian_rotation,
                self._end_effector.obj_id,
            )
        else:
            jacobian_fn = getattr(mujoco, "mj_jacBody", None)
            if jacobian_fn is None:
                jacobian_fn = getattr(mujoco, "mj_jacBodyCom", None)
            if jacobian_fn is None:
                raise RuntimeError(
                    "MuJoCo jacobian function for body end-effector is unavailable."
                )
            jacobian_fn(
                self._model,
                self._data,
                jacobian_position,
                jacobian_rotation,
                self._end_effector.obj_id,
            )
        return jacobian_position, jacobian_rotation

    def _get_end_effector_site_id(self) -> int:
        if self._end_effector.obj_type != "site":
            raise RuntimeError(
                "Operational-space control requires a site end-effector reference, "
                f"but got body '{self._end_effector.name}'."
            )
        return self._end_effector.obj_id

    def reset_robot(self) -> None:
        """Reset joints and controls to the configured home pose.

        This method is intended for reuse by task environments.
        """
        self._data.qpos[self._joint_qpos_indices] = self._home_position
        self._data.qvel[self._dof_ids] = 0.0
        self._data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        if self._model.nmocap > 0:
            end_effector_position = self._get_end_effector_position()
            end_effector_quaternion = self._get_end_effector_quaternion()
            self._data.mocap_pos[0] = end_effector_position
            self._data.mocap_quat[0] = end_effector_quaternion

    def get_gripper_pose(self) -> np.ndarray:
        """Return the current gripper control value as a 1D float array."""
        if self._gripper_actuator_id is None:
            return np.zeros((1,), dtype=np.float32)
        return np.array([self._data.ctrl[self._gripper_actuator_id]], dtype=np.float32)

    def get_robot_state(self) -> np.ndarray:
        """Return concatenated robot state vector.

        Output layout is `[joint_pos, joint_vel, gripper_pose, end_effector_pos]`.
        """
        joint_pos = self.data.qpos[self._joint_qpos_indices].astype(np.float32)
        joint_vel = self.data.qvel[self._dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()
        end_effector_pos = self._get_end_effector_position().astype(np.float32)
        return np.concatenate([joint_pos, joint_vel, gripper_pose, end_effector_pos])

    def render(self) -> list[np.ndarray]:
        """Render and return one RGB frame per configured camera."""
        frames: list[np.ndarray] = []
        camera_ids = self.camera_ids
        if not camera_ids:
            camera_ids = [cast(int, self._render_specs.camera_id)]
        for camera_id in camera_ids:
            self._viewer.update_scene(self.data, camera=camera_id)
            frames.append(self._viewer.render())
        return frames

    def apply_action(self, action: np.ndarray) -> None:
        """Apply a control action using configured `control_mode`.

        `osc` mode interprets action as Cartesian delta + optional gripper command.
        `joint` mode maps action directly to actuator controls.
        """
        action_space = cast(Any, self.action_space)
        bounded_action = np.clip(action, action_space.low, action_space.high)
        if self.control_mode == "osc":
            self._step_operational_space_control(bounded_action)
            return
        self._step_joint_control(bounded_action)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset robot-only state and return an observation dictionary."""
        super().reset(seed=seed)
        del options
        self.reset_robot()

        if self.control_mode == "osc":
            self._target_position = self._get_end_effector_position().copy()
            self._target_quaternion = self._get_end_effector_quaternion()
        mujoco.mj_forward(self._model, self._data)
        return self._compute_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step robot dynamics without task reward/termination semantics."""
        self.apply_action(action)
        observation = self._compute_observation()
        # Reward is task-specific, so we return 0.0 here. Subclasses should
        # override this method to implement reward and termination logic.
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"succeed": False}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _step_operational_space_control(self, action: np.ndarray) -> None:
        delta_position = action[:3] * 0.05
        gripper_command = 0.0
        if self._gripper_actuator_id is not None and len(action) > 6:
            gripper_command = float(action[-1])

        if self._model.nmocap > 0:
            self._data.mocap_pos[0] += delta_position
            self._data.mocap_pos[0] = np.clip(
                self._data.mocap_pos[0],
                self._cartesian_bounds[0],
                self._cartesian_bounds[1],
            )
            target_position = self._data.mocap_pos[0]
            target_quaternion = self._data.mocap_quat[0]
        else:
            self._target_position += delta_position
            self._target_position = np.clip(
                self._target_position,
                self._cartesian_bounds[0],
                self._cartesian_bounds[1],
            )
            target_position = self._target_position
            target_quaternion = self._target_quaternion

        if self._gripper_actuator_id is not None:
            current_gripper = (
                self._data.ctrl[self._gripper_actuator_id] / MAX_GRIPPER_COMMAND
            )
            target_gripper = np.clip(current_gripper + gripper_command, 0.0, 1.0)
            self._data.ctrl[self._gripper_actuator_id] = (
                target_gripper * MAX_GRIPPER_COMMAND
            )

        end_effector_site_id = self._get_end_effector_site_id()
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=end_effector_site_id,
                dof_ids=self._dof_ids,
                pos=target_position,
                ori=target_quaternion,
                joint=self._home_position,
                gravity_comp=True,
            )
            self._data.ctrl[self._actuator_ids] = tau
            mujoco.mj_step(self._model, self._data)

    def _step_joint_control(self, action: np.ndarray) -> None:
        actuator_count = len(self._actuator_ids)
        self._data.ctrl[self._actuator_ids] = action[:actuator_count]

        if self._gripper_actuator_id is not None and len(action) > actuator_count:
            self._data.ctrl[self._gripper_actuator_id] = action[actuator_count]

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def _compute_observation(self) -> dict[str, Any]:
        """Compute the current robot-centric observation."""
        joint_position = self._data.qpos[self._joint_qpos_indices].astype(np.float32)
        joint_velocity = self._data.qvel[self._dof_ids].astype(np.float32)

        end_effector_position = self._get_end_effector_position()
        end_effector_quaternion = self._get_end_effector_quaternion()
        end_effector_pose = np.concatenate(
            [end_effector_position, end_effector_quaternion]
        ).astype(np.float32)

        jac_p, jac_r = self._get_end_effector_jacobian()

        dq = self._data.qvel
        end_effector_linear_velocity = jac_p @ dq
        end_effector_angular_velocity = jac_r @ dq
        end_effector_velocity = np.concatenate(
            [end_effector_linear_velocity, end_effector_angular_velocity]
        ).astype(np.float32)

        agent_position = {
            "joint_pos": joint_position,
            "joint_vel": joint_velocity,
            "tcp_pose": end_effector_pose,
            "tcp_vel": end_effector_velocity,
            "gripper_pose": self.get_gripper_pose(),
        }

        observation: dict[str, Any] = {"agent_pos": agent_position}

        if self.image_obs and self.camera_ids:
            pixels: dict[str, np.ndarray] = {}
            for i, cam_id in enumerate(self.camera_ids):
                self._viewer.update_scene(self._data, camera=cam_id)
                pixels[self.robot_config.camera_names[i]] = self._viewer.render()
            observation["pixels"] = pixels

        return observation


class GenericTaskEnv(GenericRobotEnv):
    """Task layer built on top of `GenericRobotEnv`.

    This class adds PandaPick-like task semantics (object initialization,
    environment state in observations, reward computation, and termination)
    while retaining the generalized action handling inherited from
    `GenericRobotEnv`.
    """

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
        camera_id: int | None = None,
        reward_type: Literal["dense", "sparse"] = "sparse",
        random_object_position: bool = False,
        object_joint_name: str = "block",
        object_position_sensor_name: str = "block_pos",
        random_sampling_bounds: np.ndarray | None = None,
        object_xy_default: tuple[float, float] = (0.5, 0.0),
        lift_success_threshold: float = 0.1,
    ):
        self.reward_type = reward_type
        self._random_object_position = random_object_position
        self._object_joint_name = object_joint_name
        self._object_position_sensor_name = object_position_sensor_name
        self._object_xy_default = object_xy_default
        self._lift_success_threshold = lift_success_threshold
        self._random_sampling_bounds = (
            random_sampling_bounds
            if random_sampling_bounds is not None
            else np.asarray([[0.3, -0.15], [0.5, 0.15]], dtype=np.float64)
        )

        super().__init__(
            robot_config=robot_config,
            control_mode=control_mode,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            camera_id=camera_id,
        )

        self._object_base_height = self._resolve_object_base_height()
        self._initial_object_height = 0.0
        self._target_object_height = 0.0

        self._setup_task_observation_space()

    def _resolve_object_base_height(self) -> float:
        """Resolve object Z base from geom size when available."""
        try:
            return float(self._model.geom(self._object_joint_name).size[2])
        except Exception:
            return 0.0

    def _setup_task_observation_space(self) -> None:
        """Setup observation space compatible with task-level observations."""
        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        environment_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs and self.camera_ids:
            pixels_space: dict[str, spaces.Space[Any]] = {}
            for camera_name in self.robot_config.camera_names:
                pixels_space[camera_name] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._render_specs.height, self._render_specs.width, 3),
                    dtype=np.uint8,
                )
            self.observation_space = spaces.Dict(
                cast(
                    dict[str, Any],
                    {
                        "pixels": spaces.Dict(cast(dict[str, Any], pixels_space)),
                        "agent_pos": agent_box,
                    },
                )
            )
            return

        self.observation_space = spaces.Dict(
            cast(
                dict[str, Any],
                {
                    "agent_pos": agent_box,
                    "environment_state": environment_box,
                },
            )
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset robot and task-specific object placement state."""
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        if self._random_object_position:
            object_xy = self.random_state.uniform(*self._random_sampling_bounds)
        else:
            object_xy = np.asarray(self._object_xy_default)

        with suppress(Exception):
            self._data.jnt(self._object_joint_name).qpos[:3] = (
                float(object_xy[0]),
                float(object_xy[1]),
                self._object_base_height,
            )

        mujoco.mj_forward(self._model, self._data)

        object_position = self._read_object_position()
        self._initial_object_height = float(object_position[2])
        self._target_object_height = (
            self._initial_object_height + self._lift_success_threshold
        )

        return self._compute_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step the task environment and compute reward and termination."""
        self.apply_action(action)

        observation = self._compute_observation()
        reward = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = reward == 1.0

        object_position = self._read_object_position()
        exceeded_bounds = np.any(
            object_position[:2] < (self._random_sampling_bounds[0] - 0.05)
        ) or np.any(object_position[:2] > (self._random_sampling_bounds[1] + 0.05))

        terminated = bool(success or exceeded_bounds)
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, {"succeed": success}

    def _read_object_position(self) -> np.ndarray:
        """Read current object position from configured sensor or fallback."""
        try:
            return self._data.sensor(self._object_position_sensor_name).data.astype(
                np.float32
            )
        except Exception:
            try:
                object_joint = self._data.jnt(self._object_joint_name)
                return np.asarray(object_joint.qpos[:3], dtype=np.float32)
            except Exception:
                return np.zeros((3,), dtype=np.float32)

    def _compute_observation(self) -> dict[str, Any]:
        """Compute task observation with robot state and environment state."""
        robot_state = self.get_robot_state().astype(np.float32)
        object_position = self._read_object_position()

        if self.image_obs and self.camera_ids:
            frames = self.render()
            pixels = dict(
                zip(
                    self.robot_config.camera_names,
                    frames,
                    strict=False,
                )
            )
            return {
                "pixels": pixels,
                "agent_pos": robot_state,
            }

        return {
            "agent_pos": robot_state,
            "environment_state": object_position,
        }

    def _compute_reward(self) -> float:
        """Compute sparse/dense lifting reward from object and robot state."""
        object_position = self._read_object_position()

        if self.reward_type == "dense":
            end_effector_position = self._get_end_effector_position()
            distance = np.linalg.norm(object_position - end_effector_position)
            close_reward = np.exp(-20 * distance)
            lift_reward = (object_position[2] - self._initial_object_height) / (
                self._target_object_height - self._initial_object_height + 1e-8
            )
            lift_reward = float(np.clip(lift_reward, 0.0, 1.0))
            return float(0.3 * close_reward + 0.7 * lift_reward)

        lift = object_position[2] - self._initial_object_height
        return float(lift > self._lift_success_threshold)

    def _is_success(self) -> bool:
        """Return true when object is close to gripper and lifted enough."""
        object_position = self._read_object_position()
        end_effector_position = self._get_end_effector_position()
        distance = np.linalg.norm(object_position - end_effector_position)
        lift = object_position[2] - self._initial_object_height
        return bool(distance < 0.05 and lift > self._lift_success_threshold)
