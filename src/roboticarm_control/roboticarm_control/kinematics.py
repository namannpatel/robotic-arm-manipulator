"""Forward/inverse kinematics for the roboticarm_description arm.

Pure numpy, no ROS dependency, so it's reusable outside a running node (tests,
offline waypoint generation, future stages). Models the 5 rotational joints that
position and orient the gripper -- waist_joint, arm_1_joint, arm_2_joint,
arm_3_joint, gripper_base_joint -- exactly as chained in
``roboticarm_description/urdf/roboticArm.urdf.xacro``. The gripper's own
open/close joints (gear1_joint / gear2_joint) are not part of this chain.

The joint-axis sequence (Z, Y, Y, Z, Y) doesn't decouple into a simple closed
form, so IK is solved numerically with damped-least-squares (Levenberg-Marquardt)
on a finite-difference Jacobian -- simple to verify correct and accurate enough
for the low waypoint rate this is used at (not a real-time control-loop IK).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------- #
# Chain definition -- (joint_name, rotation axis in parent frame, origin xyz
# in parent frame), taken verbatim from roboticArm.urdf.xacro's <joint> tags.
# ---------------------------------------------------------------------- #
JOINT_NAMES = [
    "waist_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "gripper_base_joint",
]

_AXES = [
    np.array([0.0, 0.0, 1.0]),   # waist_joint
    np.array([0.0, 1.0, 0.0]),   # arm_1_joint
    np.array([0.0, 1.0, 0.0]),   # arm_2_joint
    np.array([0.0, 0.0, 1.0]),   # arm_3_joint
    np.array([0.0, 1.0, 0.0]),   # gripper_base_joint
]

_ORIGINS = [
    np.array([0.0, 0.0, 0.0]),    # waist_joint  (base_link -> waist_link)
    np.array([0.14, 0.0, 0.97]),  # arm_1_joint  (waist_link -> arm_1_link)
    np.array([0.01, 0.0, 1.2]),   # arm_2_joint  (arm_1_link -> arm_2_link)
    np.array([0.05, 0.11, 0.5]),  # arm_3_joint  (arm_2_link -> arm_3_link)
    np.array([-0.05, 0.0, 0.71]), # gripper_base_joint (arm_3_link -> gripper_base_link)
]

JOINT_LOWER = np.array([-np.pi / 2] * 5)
JOINT_UPPER = np.array([np.pi / 2] * 5)

# TCP = midpoint of the two fingertip-joint frames, expressed in
# gripper_base_link, at the gripper's neutral (fully-open==fully-closed==0)
# gear angle: gear1_joint.xyz + finger2_joint.xyz, averaged with
# gear2_joint.xyz + finger1_joint.xyz (finger1 hangs off gear2_link and vice
# versa in the source xacro).
_FINGER_VIA_GEAR1 = np.array([-0.14, 0.03, 0.37]) + np.array([-0.055, 0.162, 0.26])
_FINGER_VIA_GEAR2 = np.array([-0.12, -0.245, 0.366]) + np.array([-0.055, -0.137, 0.279])
TCP_OFFSET = 0.5 * (_FINGER_VIA_GEAR1 + _FINGER_VIA_GEAR2)


# ---------------------------------------------------------------------- #
# Rotation helpers
# ---------------------------------------------------------------------- #
def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula: rotation matrix for ``angle`` about a unit ``axis``."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    c, s = np.cos(angle), np.sin(angle)
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), robust near angle=0/pi."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near-180-degree rotation: off-diagonal formula below is ill-conditioned.
        # Fall back to extracting the axis from (R + I)/2's dominant eigenvector.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


# ---------------------------------------------------------------------- #
# Forward kinematics
# ---------------------------------------------------------------------- #
def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-frame (position, rotation) of the TCP for joint angles ``q`` (len 5)."""
    T = np.eye(4)
    for (axis, origin, angle) in zip(_AXES, _ORIGINS, q):
        step = np.eye(4)
        step[:3, :3] = rot_axis(axis, angle)
        step[:3, 3] = origin
        T = T @ step
    pos = T[:3, 3] + T[:3, :3] @ TCP_OFFSET
    return pos, T[:3, :3]


def down_orientation(azimuth: float = 0.0) -> np.ndarray:
    """Target rotation with the gripper's approach (local +Z) axis pointing
    straight down (world -Z), jaws swept out along the horizontal ``azimuth``
    (radians, 0 = world +X)."""
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


# ---------------------------------------------------------------------- #
# General numerical Jacobian + damped-least-squares IK (arbitrary 6-DOF pose
# target). Available for future stages; needs a reasonably close seed for
# targets far from it, since a single Jacobian-IK run can stall in a
# joint-limit corner. ``sample_vertical_line`` below does NOT use this -- it
# uses the reduced-chain solver, which is well-conditioned for this task by
# construction.
# ---------------------------------------------------------------------- #
def _jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """6x5 finite-difference Jacobian (top 3 rows: linear, bottom 3: angular)."""
    pos0, R0 = fk(q)
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = q.copy()
        dq[i] += eps
        pos1, R1 = fk(dq)
        J[:3, i] = (pos1 - pos0) / eps
        J[3:, i] = so3_log(R1 @ R0.T) / eps
    return J


def solve_ik(
    target_pos: np.ndarray,
    target_R: np.ndarray,
    q0: np.ndarray | None = None,
    max_iters: int = 200,
    damping: float = 0.05,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
) -> tuple[np.ndarray, bool]:
    """Damped-least-squares IK for an arbitrary target pose. Returns (q, converged)."""
    q = np.zeros(5) if q0 is None else np.array(q0, dtype=float)
    converged = False
    for _ in range(max_iters):
        pos, R = fk(q)
        e_pos = target_pos - pos
        e_rot = so3_log(target_R @ R.T)
        if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
            converged = True
            break
        e = np.concatenate([e_pos, e_rot])
        J = _jacobian(q)
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        q = np.clip(q + dq, JOINT_LOWER, JOINT_UPPER)
    return q, converged


# ---------------------------------------------------------------------- #
# Reduced-chain IK for a straight, downward-facing vertical line.
#
# waist_joint and arm_3_joint (both Z-axis) are held at 0 and don't affect
# reachability for a target directly ahead of the base (y stays whatever the
# geometry gives, constant along the line -- still a straight vertical line).
# arm_1_joint, arm_2_joint, gripper_base_joint all rotate about the parent's Y
# axis, so their rotations add: the TCP's approach axis points straight down
# (world -Z) exactly when arm_1 + arm_2 + gripper_base == -pi (verified
# against ``fk`` below). That turns "hit (x, z) while facing down" into 2
# equations in 2 unknowns (arm_1, arm_2; gripper_base is whatever keeps the
# sum at -pi) -- a small, well-conditioned Newton solve, instead of the
# general 5-DOF search which can stall in a joint-limit corner on a target
# this far from the rest pose.
# ---------------------------------------------------------------------- #
def _reduced_fk(a1: float, a2: float) -> tuple[np.ndarray, float]:
    gb = -np.pi - a1 - a2
    pos, _ = fk(np.array([0.0, a1, a2, 0.0, gb]))
    return pos, gb


def _solve_reduced_ik(
    x: float,
    z: float,
    seed_a1a2: tuple[float, float] = (-1.0, -1.0),
    max_iters: int = 100,
    tol: float = 1e-6,
    max_step: float = 0.3,
) -> tuple[np.ndarray, bool]:
    a1, a2 = seed_a1a2
    eps = 1e-6
    for _ in range(max_iters):
        pos, gb = _reduced_fk(a1, a2)
        err = np.array([x - pos[0], z - pos[2]])
        if np.linalg.norm(err) < tol:
            break
        pos_da1, _ = _reduced_fk(a1 + eps, a2)
        pos_da2, _ = _reduced_fk(a1, a2 + eps)
        J = np.array(
            [
                [(pos_da1[0] - pos[0]) / eps, (pos_da2[0] - pos[0]) / eps],
                [(pos_da1[2] - pos[2]) / eps, (pos_da2[2] - pos[2]) / eps],
            ]
        )
        try:
            delta = np.linalg.solve(J, err)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, err, rcond=None)[0]
        step_norm = np.linalg.norm(delta)
        if step_norm > max_step:
            delta *= max_step / step_norm
        a1 += delta[0]
        a2 += delta[1]

    pos, gb = _reduced_fk(a1, a2)
    q = np.array([0.0, a1, a2, 0.0, gb])
    within_limits = bool(np.all(q >= JOINT_LOWER - 1e-6) and np.all(q <= JOINT_UPPER + 1e-6))
    pos_ok = np.linalg.norm([x - pos[0], z - pos[2]]) < 1e-3
    return q, (pos_ok and within_limits)


def sample_vertical_line(
    x: float,
    z_start: float,
    z_end: float,
    n_waypoints: int = 10,
    seed_a1a2: tuple[float, float] = (-1.0, -1.0),
) -> tuple[list[np.ndarray], bool]:
    """IK-solve a straight vertical Cartesian segment: fixed x, gripper facing
    straight down throughout, z sweeping from ``z_start`` to ``z_end``. Each
    solve is seeded from the previous one (continuation) so the joint-space
    path stays smooth. Returns ``(list_of_joint_angle_arrays, all_converged)``,
    where each joint-angle array is ``[waist, arm_1, arm_2, arm_3, gripper_base]``.
    """
    a1, a2 = seed_a1a2
    waypoints = []
    all_ok = True
    for z in np.linspace(z_start, z_end, n_waypoints):
        q, ok = _solve_reduced_ik(x, z, seed_a1a2=(a1, a2))
        all_ok = all_ok and ok
        a1, a2 = q[1], q[2]
        waypoints.append(q.copy())
    return waypoints, all_ok


if __name__ == "__main__":
    # Quick standalone check: a reachable pre-grasp/grasp pair in front of the
    # base, verified against the joint limits and the down-facing constraint.
    pos0, R0 = fk(np.zeros(5))
    print(f"Rest-pose TCP: pos={pos0}, approach axis (world)={R0[:, 2]}")

    x = -1.75
    z_pre, z_grasp = 0.55, 0.05
    waypoints, ok = sample_vertical_line(x, z_pre, z_grasp, n_waypoints=8)
    print(f"converged={ok}")
    for q in waypoints:
        p, R = fk(q)
        print(
            f"q(deg)={np.degrees(q).round(1)}  tcp={p.round(3)}  "
            f"approach={R[:, 2].round(3)}"
        )
