from interbotix_xs_modules.arm import InterbotixManipulatorXS
import cv2
# from interbotix_xs_sdk.robot_manipulation import InterbotixRobot
import numpy as np
import sys

# This script makes the end-effector perform pick, pour, and place tasks
# Note that this script may not work for every arm as it was designed for the wx250
# Make sure to adjust commanded joint positions and poses as necessary
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py  # python3 bartender.py if using ROS Noetic'

def euler_to_rotation_matrix(theta, convention='Z-X-Z'):
    """
    Converts Euler angles to a rotation matrix.

    Args:
        theta: A list or array of Euler angles in radians: [roll, pitch, yaw].  
        convention: The Euler angle convention, either 'Z-X-Z' or 'X-Y-Z'.

    Returns:
        A 3x3 rotation matrix as a NumPy array.
    """

    roll, pitch, yaw = theta

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    if convention == 'Z-X-Z':
        return np.dot(Rz, np.dot(Rx, Ry))
    elif convention == 'X-Y-Z':
        return np.dot(Rz, np.dot(Ry, Rx))
    else:
        raise ValueError("Invalid Euler angle convention")
    

def main():
    bot = InterbotixManipulatorXS("rx200", "arm", "gripper")

    if (bot.arm.group_info.num_joints < 5):
        print('This demo requires the robot to have at least 5 joints!')
        sys.exit()

    # bot.arm.set_ee_pose_components(x=0.3, z=0)
    # bot.arm.set_single_joint_position("waist", np.pi/2.0)
    # bot.gripper.open()
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, z=0.16)
    # bot.gripper.close()
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, z=0.16)
    # bot.arm.set_single_joint_position("waist", -np.pi/2.0)
    # bot.arm.set_ee_cartesian_trajectory(pitch=1.5)
    # bot.arm.set_ee_cartesian_trajectory(pitch=-1.5)
    # bot.arm.set_single_joint_position("waist", np.pi/2.0)
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, z=0.16)
    # bot.gripper.open()
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, z=0.16)
    # bot.arm.go_to_home_pose()
    # bot.arm.go_to_sleep_pose()
    # bot.arm.set_ee_pose_matrix([[1,0,0, 0.2], [0,1,0, .0], [0,0,1, 0], [0,0,0,1]])
    # T = bot.arm.get_ee_pose_command()
    # print(T)
    # [[ 6.43677653e-02 -1.84130948e-02  9.97756357e-01  2.09237205e-01]
    # [ 1.38255745e-03  9.99830445e-01  1.83621788e-02  4.49421314e-03]
    # [-9.97925287e-01  1.97523074e-04  6.43823086e-02  1.68258668e-02]
    # [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    # T = np.array([[ .400961964,  1.98116958e-02,  .915880451,  .35],[-7.38164917e-03,  .999803540, -1.83954609e-02, 0],[-.916064962,  6.15171946e-04,  .401029433,  .05],[ 0,  0,  0,  1]])
    T = np.eye(4)
    euler_angles = [np.pi/2, 0, -np.pi/2]  # Example angles in radians
    convention = 'X-Y-Z'
    rotation_matrix = euler_to_rotation_matrix(euler_angles, convention)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = [0, 0.25, 0.05]
    # rotation_vector, _ = cv2.Rodrigues(T[:3,:3])
    # print(rotation_vector)
# retval, yaw, pitch, roll
    print(T)
    bot.arm.go_to_home_pose()
    bot.arm.set_ee_pose_matrix(T)
    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()