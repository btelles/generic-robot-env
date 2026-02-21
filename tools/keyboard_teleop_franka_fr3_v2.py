from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import tty
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Ensure project src is in path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generic_robot_env.generic_robot_env import GenericTaskEnv, RobotConfig


class NonBlockingKeyboard:
    def __enter__(self) -> "NonBlockingKeyboard":
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def build_env(scene_xml: Path, control_mode: str, camera_id: int) -> GenericTaskEnv:
    config = RobotConfig.from_xml(scene_xml, robot_name="franka_fr3_v2")
    return GenericTaskEnv(
        robot_config=config,
        control_mode=control_mode,
        render_mode="rgb_array",
        image_obs=False,
        camera_id=camera_id,
        reward_type="dense",
        random_object_position=False,
        object_joint_name="nonexistent_object_joint",
        object_position_sensor_name="nonexistent_object_sensor",
        random_sampling_bounds=np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float64),
        lift_success_threshold=10.0,
    )


def key_to_action(
    key: str | None,
    action_dim: int,
    linear_scale: float,
    angular_scale: float,
    gripper_scale: float,
) -> np.ndarray:
    action = np.zeros(action_dim, dtype=np.float32)
    if key is None:
        return action

    # Cartesian translation
    if key == "w" or key == ",":
        action[0] = linear_scale
    elif key == "s" or key == "o":
        action[0] = -linear_scale
    elif key == "a" or key == "a":
        action[1] = linear_scale
    elif key == "d" or key == "e":
        action[1] = -linear_scale
    elif key == "r" or key == "p":
        action[2] = linear_scale
    elif key == "f" or key == "u":
        action[2] = -linear_scale

    # Orientation deltas
    elif key == "i" or key == "c":
        action[3] = angular_scale
    elif key == "k" or key == "t":
        action[3] = -angular_scale
    elif key == "j" or key == "h":
        action[4] = angular_scale
    elif key == "l" or key == "n":
        action[4] = -angular_scale
    elif key == "u" or key == "g":
        action[5] = angular_scale
    elif key == "o" or key == "r":
        action[5] = -angular_scale

    # Gripper command (if present)
    elif action_dim > 6 and key == "z":
        action[-1] = -gripper_scale
    elif action_dim > 6 and key == "x":
        action[-1] = gripper_scale

    return action


def print_controls(action_dim: int) -> None:
    print("Keyboard teleop for GenericTaskEnv (franka_fr3_v2)")
    print("Controls:")
    print("  w/s ,/o: +x / -x")
    print("  a/d a/e: +y / -y")
    print("  r/f p/u: +z / -z")
    print("  i/k c/t: +rx / -rx")
    print("  j/l h/n: +ry / -ry")
    print("  u/o g/r: +rz / -rz")
    if action_dim > 6:
        print("  z/x: close / open gripper")
    print("  t: reset environment")
    print("  q: quit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=Path("mujoco_menagerie/franka_fr3_v2/scene.xml"),
        help="Path to FR3 scene XML.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["osc", "joint"],
        default="osc",
        help="Control mode for GenericTaskEnv.",
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--linear-scale", type=float, default=1.0)
    parser.add_argument("--angular-scale", type=float, default=0.5)
    parser.add_argument("--gripper-scale", type=float, default=0.2)
    args = parser.parse_args()

    if not args.scene_xml.exists():
        raise FileNotFoundError(f"Scene XML not found: {args.scene_xml}")

    if os.getenv("MUJOCO_GL") is None:
        os.environ["MUJOCO_GL"] = "glfw"

    env = build_env(
        scene_xml=args.scene_xml,
        control_mode=args.control_mode,
        camera_id=args.camera_id,
    )

    try:
        env.reset()
        action_dim = int(np.prod(env.action_space.shape))
        print_controls(action_dim)

        render_failed = False
        with NonBlockingKeyboard() as keyboard:
            while True:
                key = None

                if cv2 is not None and not render_failed:
                    try:
                        frame = env.render()[0]
                        cv2.imshow(
                            "franka_fr3_v2 teleop",
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        )
                        key_code = cv2.waitKey(1) & 0xFF
                        if key_code != 255:
                            key = chr(key_code)
                    except Exception as e:
                        render_failed = True
                        print(
                            f"Render preview unavailable; continuing with terminal keyboard input. Error: {e}"
                        )

                if key is None:
                    key = keyboard.read_key()

                if key == "q":
                    break
                if key == "t":
                    env.reset()
                    continue

                action = key_to_action(
                    key=key,
                    action_dim=action_dim,
                    linear_scale=args.linear_scale,
                    angular_scale=args.angular_scale,
                    gripper_scale=args.gripper_scale,
                )

                _, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(
                        f"Episode ended (reward={reward:.4f}, succeed={info.get('succeed', False)}). Resetting..."
                    )
                    env.reset()

    finally:
        env.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
