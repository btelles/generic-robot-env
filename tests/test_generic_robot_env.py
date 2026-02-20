import os

# Set headless rendering for MuJoCo
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from generic_robot_env import GenericRobotEnv, RobotConfig, extract_config_from_xml

# Minimal XML for testing
TEST_XML = """
<mujoco model="test_robot">
    <worldbody>
        <body name="base" pos="0 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.05 0.2" pos="0 0 0.2"/>
            <body name="link1" pos="0 0 0.4">
                <joint name="joint2" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.05 0.2" pos="0 0 0.2"/>
                <site name="ee" pos="0 0 0.4"/>
            </body>
        </body>
        <camera name="test_cam" pos="1 1 1" euler="0 0 0"/>
    </worldbody>
    <actuator>
        <position name="act1" joint="joint1"/>
        <velocity name="act2" joint="joint2"/>
        <!-- Reusing joint for simplicity -->
        <position name="gripper_actuator" joint="joint1"/>
    </actuator>
    <keyframe>
        <key name="home" qpos="0.1 0.2"/>
    </keyframe>
</mujoco>
"""


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_robot.xml"
    p.write_text(TEST_XML)
    return p


def test_robot_config_init() -> None:
    config = RobotConfig(
        robot_name="test",
        xml_path=Path("test.xml"),
        joint_names=["j1"],
        actuator_names=["a1"],
        ee_site_name="ee",
        camera_names=["c1"],
    )
    assert config.robot_name == "test"
    assert config.joint_names == ["j1"]


def test_extract_config_from_xml(xml_file: Path) -> None:
    config = extract_config_from_xml(xml_file, "test_robot")

    assert config.robot_name == "test_robot"
    assert "joint1" in config.joint_names
    assert "joint2" in config.joint_names
    assert "act1" in config.actuator_names
    assert config.gripper_actuator_name == "gripper_actuator"
    assert config.ee_site_name == "ee"
    assert "test_cam" in config.camera_names


def test_generic_robot_env_init(xml_file: Path) -> None:
    config = extract_config_from_xml(xml_file, "test_robot")
    env = GenericRobotEnv(robot_config=config, image_obs=True)

    obs_space = cast(Any, env.observation_space)
    assert obs_space["agent_pos"]["joint_pos"].shape == (2,)
    assert "pixels" in obs_space.spaces
    assert "test_cam" in obs_space["pixels"].spaces

    # Check action space
    # OSC mode is default. 6 (pos/ori) + 1 (gripper) = 7
    action_space = cast(Any, env.action_space)
    assert action_space.shape == (7,)

    env.close()


def test_generic_robot_env_reset(xml_file: Path) -> None:
    config = extract_config_from_xml(xml_file, "test_robot")
    env = GenericRobotEnv(robot_config=config)

    obs, _ = env.reset()

    # Check home position from keyframe
    # Joint1: 0.1, Joint2: 0.2
    np.testing.assert_allclose(obs["agent_pos"]["joint_pos"], [0.1, 0.2], atol=1e-5)

    env.close()


def test_generic_robot_env_step_osc(xml_file: Path) -> None:
    config = extract_config_from_xml(xml_file, "test_robot")
    env = GenericRobotEnv(robot_config=config, control_mode="osc")
    env.reset()

    # Step with zero action (mostly)
    action_space = cast(Any, env.action_space)
    action = np.zeros(action_space.shape, dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    assert "agent_pos" in obs
    assert obs["agent_pos"]["tcp_pose"].shape == (7,)

    env.close()


def test_generic_robot_env_step_joint(xml_file: Path) -> None:
    config = extract_config_from_xml(xml_file, "test_robot")
    env = GenericRobotEnv(robot_config=config, control_mode="joint")
    env.reset()

    # 2 actuators (act1, act2) + 1 gripper = 3 actions
    action_space = cast(Any, env.action_space)
    assert action_space.shape == (3,)

    action = np.array([0.5, -0.5, 0.1], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    # Check if simulation moved (non-zero velocity most likely after step)
    assert np.any(obs["agent_pos"]["joint_vel"] != 0)

    env.close()


def test_generic_robot_env_render(xml_file: Path) -> None:
    from gym_hil.mujoco_gym_env import GymRenderingSpec

    config = extract_config_from_xml(xml_file, "test_robot")
    render_spec = GymRenderingSpec(width=640, height=480)
    env = GenericRobotEnv(
        robot_config=config,
        image_obs=True,
        render_mode="rgb_array",
        render_spec=render_spec,
    )
    env.reset()

    action_space = cast(Any, env.action_space)
    obs, _, _, _, _ = env.step(action_space.sample())
    assert "pixels" in obs
    assert "test_cam" in obs["pixels"]
    assert obs["pixels"]["test_cam"].shape == (480, 640, 3)

    env.close()
