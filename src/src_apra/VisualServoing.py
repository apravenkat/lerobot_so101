import cv2
import numpy as np
import os
import mujoco
import mujoco.viewer as mujoco_viewer
from perception_stack import PerceptionStack
from motion_planner import MotionPlanner
from ObjectLocaloization import ObjectLocalization
import threading
import time
class VisualSeroving:
    def __init__(self):
        self.ol = ObjectLocalization()

    def servo_to_object(self):
        
        pos, rot = self.ol.getObjectPosition()
        if pos is None or rot is None:
            print("Object not found.")
            return
        print(f"Object Position: {pos}")
        
        return pos
    
    def track_object(self):
        current_joints = self.ol.planner.getRobotPose()
        prev_error = 0
        self.ol.planner.data.qpos[:len(self.ol.planner.joint_names)] = np.radians(current_robot_joint_positions)
        mujoco.mj_forward(self.ol.planner.model, self.ol.planner.data)
        current_end_effector_pos = self.ol.planner.getEndEffectorPosition()
        site_id = mujoco.mj_name2id(self.ol.planner.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        nq = len(self.ol.planner.joint_names)
        nv = self.ol.planner.model.nv
        while True:
            frame = self.ol.perception.get_frame()
            if frame is not None:
                pos = self.servo_to_object() #end-effector position to be servoed to in world coordinates wrt robot base
                if pos is not None:
                        current_end_effector_pos = self.ol.planner.getEndEffectorPosition()
                        error = pos - current_end_effector_pos
                        J_pos = np.zeros((3, nv))
                        J_rot = np.zeros((3, nv))
                        mujoco.mj_jacSite(self.ol.planner.model, self.ol.planner.data, J_pos, J_rot, site_id)
                        J_pos = J_pos[:, :nq]
                        q_dot = self.ol.planner.data.qvel[:nq]
                        ee_vel = J_pos @ q_dot
                        vel_err = -ee_vel
                        lambda_gain = 3
                        d_theta = lambda_gain * (J_pos.T @ error) 
                        max_step = np.radians(3.0)   # example: limit to 2° per cycle
                        d_theta = np.clip(d_theta, -max_step, max_step)
                        current_joints = self.ol.planner.getRobotPose()
                        new_joints = current_joints + np.degrees(d_theta)
                        self.ol.planner.sendSimPoseToRobot(new_joints)
                        self.ol.sync_robot_and_simulation() #sync mujoco robot positions with real robot
                        mujoco.mj_forward(self.ol.planner.model, self.ol.planner.data)
   
            time.sleep(0.001)  # Adjust the sleep time as needed to control the tracking frequency
        
if __name__ == "__main__":
    visual_servoing = VisualSeroving()
    current_robot_joint_positions = visual_servoing.ol.planner.getRobotPose()
    visual_servoing.ol.planner.data.qpos[:len(visual_servoing.ol.planner.joint_names)] = np.radians(current_robot_joint_positions)
    print("Current Robot Joint Positions (radians):", np.radians(current_robot_joint_positions))
    site_id = mujoco.mj_name2id(visual_servoing.ol.planner.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    mujoco.mj_forward(visual_servoing.ol.planner.model, visual_servoing.ol.planner.data)
    home_pos = np.array([-1.758, -79.604, 40.483, 45.582, -91.120, 0.395])
    visual_servoing.ol.goto_home_position(home_pos)
    print("Do you want to localize the object? (y/n)")
    choice = input().strip().lower()
    if choice != 'y':
        print("Object localization cancelled.")
        visual_servoing.ol.planner.arm.freeMoveRobot()
        exit(0)
    visual_servoing.ol.go_to_object()
    print("Do you want to start visual servoing to track the object? (y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        visual_servoing.track_object()
    else:
        print("Visual servoing cancelled.")
        visual_servoing.ol.planner.arm.freeMoveRobot()