"""Raw-keypress teleop: press d / g / u to step through the pick sequence.

Stdlib-only (tty/termios) -- no pynput, which also isn't installed here.
Publishes one std_msgs/String per keypress on /pick_stage_cmd; motion_node
decides whether the command is valid right now (see STAGE_ORDER there).
Run this in its own terminal after the sim launch is up:
    ros2 run roboticarm_control stage_teleop
"""

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

KEY_MAP = {
    "d": "down",
    "g": "grasp",
    "u": "up",
}

INSTRUCTIONS = """
Pick-sequence teleop
  d = move gripper down
  g = grasp (close gripper)
  u = move gripper up
  q = quit
Steps run strictly in order (down -> grasp -> up); out-of-order or repeated
keys are ignored (see motion_node's log for why).
"""


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = Node("stage_teleop")
    pub = node.create_publisher(String, "/pick_stage_cmd", 10)

    print(INSTRUCTIONS)
    try:
        while rclpy.ok():
            key = _read_key()
            if key.lower() == "q":
                break
            cmd = KEY_MAP.get(key.lower())
            if cmd is None:
                continue
            msg = String()
            msg.data = cmd
            pub.publish(msg)
            print(f"-> published '{cmd}'")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
