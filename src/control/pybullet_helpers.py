import pybullet as p
import numpy as np
import time

def move_arm_to(kuka_id, num_joints, target_pos, force=10000, max_velocity=100):
    """
    move arm to pos using inverse kinematics
    """
    joint_positions = p.calculateInverseKinematics(kuka_id, 6, target_pos)
    for j in range(num_joints):
        p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, joint_positions[j], force=force, maxVelocity=max_velocity)


def wait_for_arm_to_reach(kuka_id, target_pos, threshold=0.1):
    """
    returns true if end effector is close enogh to target pos
    """
    ee_pos = p.getLinkState(kuka_id, 6)[0]
    dist = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
    return dist < threshold


def grab_object(kuka_id, ball_id):
    """
    grabs object with end effector
    """
    constraint_id = p.createConstraint(
        kuka_id, 6, ball_id, -1, p.JOINT_FIXED,
        [0, 0, 0], [0, 0, 0], [0, 0, 0]
    )
    return constraint_id


def release_object(constraint_id):
    """
    releases object
    """
    p.removeConstraint(constraint_id)


def move_arm_to_joint_positions(kuka_id, num_joints, target_joint_positions, force=10000, max_velocity=100):
    """
    move kuka arm to static pos by manually setting joint angles
    
    Args:
        kuka_id: PyBullet body ID of the KUKA robot
        num_joints: Number of joints to control
        target_joint_positions: List of target joint angles in radians
        force: Force to apply to joints
        max_velocity: Maximum velocity for joint movement
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
    basically a threshold checker for joint positions
    
    Args:
        kuka_id: PyBullet body ID of the KUKA robot
        target_joint_positions: List of target joint angles in radians
        threshold: Maximum allowed difference in radians
        
    Returns:
        bool: True if all joints are within threshold of their targets
    """
    for joint_idx in range(len(target_joint_positions)):
        current_pos = p.getJointState(kuka_id, joint_idx)[0]
        target_pos = target_joint_positions[joint_idx]
        if abs(current_pos - target_pos) > threshold:
            return False
    return True


def get_initial_joint_positions(kuka_id, num_joints):
    """
    gets current joint positions
    
    Args:
        kuka_id: PyBullet body ID of the KUKA robot
        num_joints: Number of joints to read
        
    Returns:
        list: Current joint positions in radians
    """
    joint_positions = []
    for joint_idx in range(num_joints):
        joint_state = p.getJointState(kuka_id, joint_idx)
        joint_positions.append(joint_state[0])  # Joint position (angle)
    return joint_positions 