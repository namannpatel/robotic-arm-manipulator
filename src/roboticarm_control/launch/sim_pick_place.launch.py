"""Bring up the arm in Gazebo Fortress with ros2_control and the pick-stage motion node.

Uses Gazebo Fortress (ros_gz_sim / gz_ros2_control), not Gazebo Classic --
Classic's gzserver deadlocks during Ogre1 render-thread init on this machine
(reproducible even under Xvfb / forced software rendering / restricted CPU
affinity), while Fortress's Ogre2-based gz-sim starts cleanly here.

Starts gz-sim headless (empty world), spawns the robot from a xacro-processed
robot_description plus a small box as a visual pick target, bridges /clock so
ros2_control gets simulation time, then chains the controller spawners off the
process-exit events of the steps they depend on (spawn -> joint_state_broadcaster
-> arm_controller/gripper_controller) so nothing races the controller_manager
coming up (started inside the gz-sim process by the gz_ros2_control plugin).
Each spawn point also gets a short settle delay: the controller_manager's
'/controller_manager/list_controllers' service appears (so spawner's own
wait_for_service passes) a beat before gz_ros2_control has actually built the
hardware interfaces from the first simulation step, so calling load_controller
right when the service appears intermittently fails with "Failed loading
controller" -- a couple of seconds fixes it reliably.
Run `ros2 run roboticarm_control stage_teleop` in a second terminal afterwards
to drive it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_pkg = get_package_share_directory("roboticarm_description")
    control_pkg = get_package_share_directory("roboticarm_control")
    ros_gz_sim_pkg = get_package_share_directory("ros_gz_sim")

    xacro_file = os.path.join(description_pkg, "urdf", "roboticArm.urdf.xacro")
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_pkg, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": "-r -s empty.sdf"}.items(),
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "roboticarm"],
    )

    spawn_target = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file", os.path.join(control_pkg, "worlds", "pick_target.sdf"),
            "-name", "pick_target",
            "-x", "-1.75", "-y", "0.015", "-z", "0.1",
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen",
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller"],
        output="screen",
    )

    motion_node = Node(
        package="roboticarm_control",
        executable="motion_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            gz_sim,
            clock_bridge,
            robot_state_publisher,
            spawn_robot,
            spawn_target,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawn_robot,
                    on_exit=[TimerAction(period=3.0, actions=[joint_state_broadcaster_spawner])],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_broadcaster_spawner,
                    on_exit=[
                        TimerAction(
                            period=2.0,
                            actions=[arm_controller_spawner, gripper_controller_spawner],
                        )
                    ],
                )
            ),
            motion_node,
        ]
    )
