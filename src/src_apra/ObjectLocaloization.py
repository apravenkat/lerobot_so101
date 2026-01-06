import cv2
import numpy as np
import time
import os
import mujoco
import mujoco.viewer
from motion_planner import MotionPlanner
from perception_stack import PerceptionStack
from scipy.spatial.transform import Rotation as R


class ObjectLocalization:
  
   def __init__(self, hand_eye_calib_file='handeye_calibration_chessboard.npz', camera_calib_file='camera_calibration_data.npz'):
       base_dir = os.path.dirname(os.path.abspath(__file__))
       self.perception = PerceptionStack()
       self.planner = MotionPlanner()
       self.hand_eye_calib = np.load(os.path.join(base_dir, "calibrations",hand_eye_calib_file))
       self.camera_calib = np.load(os.path.join(base_dir, "calibrations", camera_calib_file))
       self.camera_matrix = self.camera_calib['camera_matrix']
       self.dist_coeffs = self.camera_calib['dist_coeffs']
       self.T_cam2gripper = self.hand_eye_calib['T_cam2gripper']
       self.length = 0.04  # Marker length in meters
      
  
   def localize_object(self, frame):
  
       aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
       parameters = cv2.aruco.DetectorParameters()
       detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


       corners, ids, _ = detector.detectMarkers(frame)
       if ids is None or len(ids) == 0:
           print("No ArUco markers detected.")
           return None


       success, rvec, tvec = cv2.solvePnP(
           np.array([
               [-self.length/2,  self.length/2, 0],
               [ self.length/2,  self.length/2, 0],
               [ self.length/2, -self.length/2, 0],
               [-self.length/2, -self.length/2, 0],
           ], dtype=np.float32),
           corners[0][0],
           self.camera_matrix,
           self.dist_coeffs
       )


       if not success:
           print("Pose estimation failed.")
           return None
       offset = np.array([0, 0.05, 0.05,1 ])
       R_cam2obj, _ = cv2.Rodrigues(rvec)
       T_cam2obj = np.eye(4)
       T_cam2obj[:3, :3] = R_cam2obj
       T_cam2obj[:3, 3] = tvec.ravel()
      
      
       #T_gripper2obj = np.linalg.inv(self.T_cam2gripper) @ np.linalg.inv(T_cam2obj)
       T_base2gripper =(self.planner.get_base2gripper_transform())
       T_base2obj = T_base2gripper @ (self.T_cam2gripper) @ (T_cam2obj)
       r_offset = T_base2obj @ offset
       T_base2obj[:3, 3] = r_offset[:3]
       return T_base2obj
  


   def getObjectPosition(self):
       while True:
           frame = self.perception.get_frame()
          
          
           if frame is None:
               print("No frame captured.")
               time.sleep(0.001)
               continue
           self.sync_robot_and_simulation()
           T_base2obj = self.localize_object(frame)
           if T_base2obj is not None:
               pos = T_base2obj[:3, 3].copy()
               rot = T_base2obj[:3, :3].copy()
               print(f"Object Position in Base Frame: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
               return pos, rot
          
           else:
               print("Object not localized.")
      


  
   def sync_robot_and_simulation(self):
       current_robot_joint_positions = self.planner.getRobotPose()
       self.planner.data.qpos[:len(self.planner.joint_names)] = np.radians(current_robot_joint_positions)
       site_id = mujoco.mj_name2id(self.planner.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")  
       mujoco.mj_forward(self.planner.model, self.planner.data)
  
   def visualize_localization(self):
      
       # Define the index for the user geometry (must be < mjMAXUIUSERGEOM, default is 100)
       GEOM_ID = 0
      
       with mujoco.viewer.launch_passive(self.planner.model, self.planner.data) as viewer:
          
           # --- 1. INITIALIZE GEOMETRY ONCE ---
           # NOTE: We access the scene data via 'viewer.user_scn', not 'self.planner.data'.
          
           # The available slots is the size of the pre-allocated array.
           # We ensure the GEOM_ID is valid before proceeding.
           if GEOM_ID >= viewer.user_scn.maxgeom:
                print(f"Error: Geometry ID {GEOM_ID} is out of bounds for the user scene (max: {viewer.user_scn.maxgeom - 1}).")
                return
          
           # Initialize the geometry properties
           mujoco.mjv_initGeom(
               viewer.user_scn.geoms[GEOM_ID],
               type=mujoco.mjtGeom.mjGEOM_SPHERE,
               size=[0.02, 0, 0],  # Radius of the sphere
               pos=np.array([0, 0, 0]), # Initial position
               mat=np.eye(3, dtype=np.float64).flatten(), # Initial orientation (identity)
               rgba=np.array([1, 0, 0, 1]) # Red color
           )
          
           # Crucially, update the number of active user geometries.
           # We use max() to ensure we don't accidentally shrink the scene count.
           # viewer.user_scn.ngeom tells the renderer how many of the 'geoms' array slots to draw.
           viewer.user_scn.ngeom = max(viewer.user_scn.ngeom, GEOM_ID + 1)
          
           while True:
               frame = self.perception.get_frame()
               cv2.imshow('Camera Frame', frame)
              
               if frame is None:
                   print("No frame captured.")
               if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                   break
               geom = viewer.user_scn.geoms[GEOM_ID]
               T_base2obj = self.localize_object(frame)
              
               if T_base2obj is not None:
                   pos = T_base2obj[:3, 3]
                   rot = T_base2obj[:3, :3]
                  
                   # Convert 3x3 rotation matrix to 9-element flattened array (column-major)
                   #geom.mat[:] = rot.T.flatten() 
                  
                   # --- 2. UPDATE GEOMETRY POSE IN THE LOOP ---
                  
                   geom.pos[:] = pos # Use slice assignment for numpy array for efficiency
                   geom.mat[:] = rot.T # Use slice assignment for numpy array for efficiency
               self.sync_robot_and_simulation()
               mujoco.mj_forward(self.planner.model, self.planner.data)
               mujoco.mj_step(self.planner.model, self.planner.data)
               viewer.sync()
               time.sleep(0.01)
   def goto_home_position(self, home_pos):
       home_pos = np.radians(home_pos)
       traj = self.planner.plan_motion_to_joint_positions(home_pos)
       if traj is not None:
           self.planner.visualize_trajectory(traj, timestep=0.05)
           print("do you want to move the robot to home position? (y/n)")
           choice = input().strip().lower()
           if choice != 'y':
               print("Motion execution cancelled.")
               return
           else:
               for qpos in traj:
                   smoothed_qpos = self.planner.smooth_joint_positions(np.degrees(qpos[:len(self.planner.joint_names)]))
                   self.planner.sendSimPoseToRobot(smoothed_qpos)
                   time.sleep(0.01)
           print("Robot moved to home position.")
       else:
           print("Failed to plan motion to home position.")
   def go_to_object(self):
       pos, rot = self.getObjectPosition()
      
       R_me = np.eye(3)
       theta = np.pi/2
       #R_me[0,0] = -1
       #R_me[2,2] = -1
       R_me = np.array([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])
       rot = rot @ R_me
       r = R.from_matrix(rot)
       roll, pitch, yaw = np.degrees(r.as_euler('xyz', degrees=False))
       #rot = None
       print(f"Object Rotation (radians): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
       traj = self.planner.plan_motion(pos, rot)
       if traj is None:
           print("Failed to find a motion plan.")
       else:
           print("Motion plan completed.")
           self.planner.visualize_trajectory(traj, timestep=0.05)
           print("Visualization completed.")
           print("Do you want to execute this motion on the real robot? (y/n)")
           choice = input().strip().lower()
           if choice != 'y':
               print("Motion execution cancelled.")
               return
           for qpos in traj:
               smoothed_qpos = self.planner.smooth_joint_positions(np.degrees(qpos[:len(self.planner.joint_names)]))
               self.planner.sendSimPoseToRobot(smoothed_qpos)
               time.sleep(0.01)
           print("Motion execution completed.")
           del pos, rot, traj
if __name__ == "__main__":
   locator = ObjectLocalization()
   current_robot_joint_positions = locator.planner.getRobotPose()
   locator.planner.data.qpos[:len(locator.planner.joint_names)] = np.radians(current_robot_joint_positions)
   print("Current Robot Joint Positions (radians):", np.radians(current_robot_joint_positions))
   site_id = mujoco.mj_name2id(locator.planner.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
   mujoco.mj_forward(locator.planner.model, locator.planner.data)
   #locator.visualize_localization()
   home_pos = np.array([-1.758, -79.604, 40.483, 45.582, -91.120, 0.395])
   while(True):
       print("Do you want to move the robot to home position? (y/n)")
       choice = input().strip().lower()
       if choice == 'y':
           time.sleep(1)
           locator.goto_home_position(home_pos)
      
       print("Do you want to localize the object? (y/n)")
       choice = input().strip().lower()
       if choice == 'y':
           time.sleep(1)
           locator.go_to_object()
       print("Do you want to exit? (y/n)")
       choice = input().strip().lower()
       if choice == 'y':
           locator.planner.arm.freeMoveRobot()
           break

