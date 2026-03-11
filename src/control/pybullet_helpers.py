import pybullet as p
import numpy as np
import time

def move_arm_to(kuka_id, num_joints, target_pos, force=10000, max_velocity=100):
    """
    moves arm using IK to target pos
    """
    joint_positions = p.calculateInverseKinematics(kuka_id, 6, target_pos)
    for j in range(num_joints):
        p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, joint_positions[j], force=force, maxVelocity=max_velocity)


def wait_for_arm_to_reach(kuka_id, target_pos, threshold=0.1):
    """
    checks if ee is close enough to target pos
    """
    ee_pos = p.getLinkState(kuka_id, 6)[0]
    dist = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
    return dist < threshold


def grab_object(kuka_id, ball_id):
    """
    grabs obj with fixed constraint
    """
    constraint_id = p.createConstraint(
        kuka_id, 6, ball_id, -1, p.JOINT_FIXED,
        [0, 0, 0], [0, 0, 0], [0, 0, 0]
    )
    return constraint_id


def release_object(constraint_id):
    """releases constraint"""
    p.removeConstraint(constraint_id)


def move_arm_to_joint_positions(kuka_id, num_joints, target_joint_positions, force=10000, max_velocity=100):
    """
    moves arm by setting joint angles directly (FK)
    """
    for joint_idx in range(min(num_joints, len(target_joint_positions))):
        p.setJointMotorControl2(
            kuka_id, 
            joint_idx, 
            p.POSITION_CONTROL, 
            target_joint_positions[joint_idx], 
            force=force, 
            maxVelocity=max_velocity
        )


def wait_for_joints_to_reach(kuka_id, target_joint_positions, threshold=0.1):
    """
    check if joints reached target
    """
    for joint_idx in range(len(target_joint_positions)):
        current_pos = p.getJointState(kuka_id, joint_idx)[0]
        target_pos = target_joint_positions[joint_idx]
        if abs(current_pos - target_pos) > threshold:
            return False
    return True


def get_initial_joint_positions(kuka_id, num_joints):
    """
    get current joint positions (for reset)
    """
    joint_positions = []
    for joint_idx in range(num_joints):
        joint_state = p.getJointState(kuka_id, joint_idx)
        joint_positions.append(joint_state[0])
    return joint_positions
