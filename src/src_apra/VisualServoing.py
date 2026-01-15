import cv2
import numpy as np
import os
import mujoco
import mujoco.viewer as mujoco_viewer
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
        self.ol.sync_robot_and_simulation()
        mujoco.mj_forward(self.ol.planner.model, self.ol.planner.data)
        current_end_effector_pos = self.ol.planner.getEndEffectorPosition()
        site_id = mujoco.mj_name2id(self.ol.planner.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        nq = len(self.ol.planner.joint_names)
        nv = self.ol.planner.model.nv
        print("NQ:", nq, "NV:", nv)
        total_error = 0
        integral_gain = 0.0001
        p_gain = 2.5
        gain = 0.1
        damping = 0.15
        lambda_ = 0.1
        d_gain = 0.001
        control_type = 2
        prev_error = 0
        self.ol.sync_robot_and_simulation() #sync mujoco robot positions with real robot
        try:
            while True:
                frame = self.ol.perception.get_frame()
                cv2.imshow('Camera Frame', frame)
                cv2.waitKey(1)
                if frame is not None:
                    pos = self.servo_to_object() #end-effector position to be servoed to in world coordinates wrt robot base
                    if pos is not None:
                        current_end_effector_pos, current_end_effector_rot = self.ol.planner.getEndEffectorPosition()
                        error = pos - current_end_effector_pos
                        J_pos = np.zeros((3, nv))
                        J_rot = np.zeros((3, nv))
                        mujoco.mj_jacSite(self.ol.planner.model, self.ol.planner.data, J_pos, J_rot, site_id)
                        q_dot = self.ol.planner.data.qvel[:nq]
                        if control_type==1:
                            q_dot = self.ol.planner.data.qvel[:nq]
                            ee_vel = J_pos @ q_dot
                            vel_err = -ee_vel
                            angle_error = J_pos.T @ error
                            total_error += angle_error
                            total_error = np.clip(total_error,-1,1)
                            d_theta = p_gain * (angle_error) + integral_gain * total_error + d_gain * (angle_error-prev_error)

                        elif control_type==2:
                            J_pos_T = J_pos.T    
                            lambda_sq = (lambda_) * np.eye(3)
                            JJt = J_pos @ J_pos_T + lambda_sq
                            d_theta =  J_pos_T @ np.linalg.solve(JJt, error) 
                            max_step = np.radians(2)
                            d_theta = np.clip(d_theta, -max_step, max_step)
                            print("d_theta:", d_theta)
                        current_joints = self.ol.planner.getRobotPose()
                        new_joints = current_joints + (np.degrees(d_theta))
                        #self.ol.planner.smooth_joint_positions(new_joints)
                        self.ol.planner.sendSimPoseToRobot(new_joints)
                        self.ol.planner.data.qpos[:nq] = np.radians(new_joints)
                        mujoco.mj_forward(self.ol.planner.model, self.ol.planner.data)
                        #prev_error = angle_error
                        time.sleep(0.01)
        except KeyboardInterrupt:
            print("Visual servoing stopped by user.")
            self.ol.planner.arm.freeMoveRobot()   

            
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