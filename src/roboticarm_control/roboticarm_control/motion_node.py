"""Stage-driven pick motion: down / grasp / up, triggered over /pick_stage_cmd.

Straight vertical Cartesian waypoints (fixed x, gripper facing straight down)
come from :mod:`roboticarm_control.kinematics`; each stage is sent as one
FollowJointTrajectory goal to the ros2_control controllers defined in
config/controllers.yaml. Commands are accepted strictly in order
(down -> grasp -> up -> down -> ...); anything out of order, or arriving while
a goal is still executing, is logged and ignored -- this is the "one stage
finishes before the next key does anything" behavior that was asked for.
"""

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from roboticarm_control.kinematics import sample_vertical_line

ARM_JOINTS = ["waist_joint", "arm_1_joint", "arm_2_joint", "arm_3_joint", "gripper_base_joint"]
GRIPPER_JOINTS = ["gear1_joint", "gear2_joint"]

STAGE_ORDER = ["down", "grasp", "up"]


class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        self.declare_parameter("target_x", -1.75)
        self.declare_parameter("pregrasp_z", 0.55)
        self.declare_parameter("grasp_z", 0.05)
        self.declare_parameter("n_waypoints", 10)
        self.declare_parameter("segment_duration_s", 4.0)
        self.declare_parameter("gripper_open_pos", 0.0)
        self.declare_parameter("gripper_closed_pos", 0.6)
        self.declare_parameter("grasp_duration_s", 1.5)

        x = self.get_parameter("target_x").value
        z_pre = self.get_parameter("pregrasp_z").value
        z_grasp = self.get_parameter("grasp_z").value
        n_wp = self.get_parameter("n_waypoints").value
        self._segment_duration = self.get_parameter("segment_duration_s").value
        self._gripper_open = self.get_parameter("gripper_open_pos").value
        self._gripper_closed = self.get_parameter("gripper_closed_pos").value
        self._grasp_duration = self.get_parameter("grasp_duration_s").value

        down_waypoints, ok = sample_vertical_line(x, z_pre, z_grasp, n_waypoints=n_wp)
        if not ok:
            self.get_logger().error(
                "IK did not fully converge for the requested pick line "
                f"(x={x}, z: {z_pre}->{z_grasp}); trajectory will still be sent "
                "but the arm may not track it exactly. Check the joint limits "
                "against these parameters."
            )
        self._down_waypoints = down_waypoints
        self._up_waypoints = list(reversed(down_waypoints))

        self._arm_client = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )

        self._busy = False
        self._next_index = 0  # index into STAGE_ORDER

        self.create_subscription(String, "/pick_stage_cmd", self._on_cmd, 10)
        self.get_logger().info(
            "motion_node ready. Waiting for /pick_stage_cmd: "
            f"expecting '{STAGE_ORDER[self._next_index]}' next."
        )

    # ------------------------------------------------------------------ #
    def _on_cmd(self, msg: String):
        cmd = msg.data.strip().lower()
        expected = STAGE_ORDER[self._next_index]

        if self._busy:
            self.get_logger().warn(f"Ignoring '{cmd}': previous stage still executing.")
            return
        if cmd != expected:
            self.get_logger().warn(f"Ignoring '{cmd}': expecting '{expected}' next.")
            return

        if cmd == "down":
            self._send_arm_trajectory(self._down_waypoints, "down")
        elif cmd == "grasp":
            self._send_gripper_goal(self._gripper_closed, "grasp")
        elif cmd == "up":
            self._send_arm_trajectory(self._up_waypoints, "up")

    def _advance(self):
        self._next_index = (self._next_index + 1) % len(STAGE_ORDER)
        self.get_logger().info(f"Stage done. Expecting '{STAGE_ORDER[self._next_index]}' next.")

    # ------------------------------------------------------------------ #
    def _send_arm_trajectory(self, waypoints, label):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        n = len(waypoints)
        for i, q in enumerate(waypoints):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            t = self._segment_duration * (i + 1) / n
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            goal.trajectory.points.append(pt)
        self._send_goal(self._arm_client, goal, label)

    def _send_gripper_goal(self, position, label):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(position), float(-position)]  # gear2 mirrors gear1
        pt.time_from_start.sec = int(self._grasp_duration)
        pt.time_from_start.nanosec = int((self._grasp_duration - int(self._grasp_duration)) * 1e9)
        goal.trajectory.points.append(pt)
        self._send_goal(self._gripper_client, goal, label)

    def _send_goal(self, client: ActionClient, goal, label: str):
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"'{label}': action server not available, dropping command.")
            return
        self._busy = True
        self.get_logger().info(f"Sending '{label}' trajectory.")
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_goal_response(f, label))

    def _on_goal_response(self, future, label):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"'{label}' goal rejected.")
            self._busy = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_result(f, label))

    def _on_result(self, future, label):
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"'{label}' finished.")
        else:
            self.get_logger().warn(f"'{label}' finished with status={result.status}.")
        self._busy = False
        self._advance()


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
