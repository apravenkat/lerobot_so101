import cv2
import numpy as np
import time
import os
import mujoco
import mujoco.viewer
from motion_planner import MotionPlanner
from perception_stack import PerceptionStack  # your existing perception class


class HandEyeCalibration:
    def __init__(self, camera_matrix_path='camera_calibration_data.npz'):
        # Initialize components
        self.perception = PerceptionStack()
        self.motion_planner = MotionPlanner()
        self.motion_planner.arm.freeMoveRobot()

        # Load camera calibration
        data = np.load(camera_matrix_path)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']

        # Storage for calibration pairs
        self.robot_poses = []   # list of 4x4 transformation matrices (end-effector in base)
        self.camera_poses = []  # list of 4x4 transformation matrices (board in camera)

        # Checkerboard parameters
        self.pattern_size = (8, 5)  # inner corners (columns, rows)
        self.square_size = 0.015    # meters

        # Precompute 3D object points for the checkerboard (z = 0 plane)
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.objp = objp * self.square_size

    def capture_handeye_pair(self):
        """Capture one pair of (robot pose, camera pose) for hand-eye calibration."""
        frame = self.perception.get_frame()
        if frame is None:
            print("No frame captured.")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect checkerboard corners with subpixel refinement
        found, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
        if not found:
            print("Checkerboard not detected.")
            return False

        # Refine corner positions to subpixel accuracy
        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            (5, 5),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # Estimate pose of the checkerboard (board in camera frame)
        success, rvec, tvec = cv2.solvePnP(
            self.objp,
            corners_subpix,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("Pose estimation failed.")
            return False
        projected_points, _ = cv2.projectPoints(self.objp, rvec, tvec, self.camera_matrix, self.dist_coeffs) 
        error = np.linalg.norm(corners_subpix.reshape(-1,2) - projected_points.reshape(-1,2), axis=1) 
        mean_error = np.mean(error) 
        print(f"Reprojection error: {mean_error:.2f} px") 
        R_board2cam, _ = cv2.Rodrigues(rvec)
        T_board2cam = np.eye(4)
        T_board2cam[:3, :3] = R_board2cam
        T_board2cam[:3, 3] = tvec.ravel()

        # Get robot end-effector pose in base frame (from MuJoCo)
        current_robot_joint_positions = self.motion_planner.getRobotPose()
        self.motion_planner.data.qpos[:len(self.motion_planner.joint_names)] = np.radians(current_robot_joint_positions)
        mujoco.mj_forward(self.motion_planner.model, self.motion_planner.data)

        site_id = mujoco.mj_name2id(
            self.motion_planner.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        R_base2ee = self.motion_planner.data.site_xmat[site_id].reshape(3, 3)
        t_base2ee = self.motion_planner.data.site_xpos[site_id]
        T_base2ee = np.eye(4)
        T_base2ee[:3, :3] = R_base2ee
        T_base2ee[:3, 3] = t_base2ee

        print("Robot EE position:", t_base2ee)
        print("Checkerboard tvec (camera frame):", tvec.ravel())

        # Store for hand–eye calibration
        self.camera_poses.append(T_board2cam)
        self.robot_poses.append(T_base2ee)

        print(f"Captured pair #{len(self.robot_poses)}")
        cv2.drawChessboardCorners(frame, self.pattern_size, corners_subpix, found)
        for p in projected_points:
            cv2.circle(frame, tuple(p.ravel().astype(int)), 3, (0,0,255), -1)  # red reprojections
        cv2.imshow('Frame', frame)
        cv2.waitKey(500)
        return True

    def run_calibration(self, num_samples=10):
        """Guide user through capturing calibration samples."""
        print("Starting Hand-Eye Calibration (Checkerboard version)")
        print("Move the robot to different poses and press [c] to capture or [q] to quit.")
        os.makedirs("handeye_data", exist_ok=True)

        while len(self.robot_poses) < num_samples:
            frame = self.perception.get_frame()
            if frame is not None:
                cv2.imshow("Camera Feed", frame)

            key = cv2.waitKey(100) & 0xFF
            if key == ord('c'):
                result = self.capture_handeye_pair()
                print(result)
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"Captured {len(self.robot_poses)} valid pairs.")
        if len(self.robot_poses) < 3:
            print("Not enough data for calibration.")
            return

        self.compute_handeye_transform()

    def compute_handeye_transform(self):
        """Compute hand-eye using relative motions (AX = XB)."""
        if len(self.robot_poses) < 2 or len(self.camera_poses) < 2:
            print("Need at least 2 poses to compute relative motions.")
            return

        A_R, A_t, B_R, B_t = [], [], [], []
        n = len(self.robot_poses)
        for i in range(n):
            A =  self.robot_poses[i]
            B = self.camera_poses[i]
            A_R.append(A[:3, :3])
            A_t.append(A[:3, 3])
            B_R.append(B[:3, :3])
            B_t.append(B[:3, 3])

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            A_R, A_t, B_R, B_t, method=cv2.CALIB_HAND_EYE_TSAI
        )

        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.ravel()

        np.savez(
            "handeye_data/handeye_calibration_chessboard.npz",
            T_cam2gripper=T_cam2gripper,
            R=R_cam2gripper,
            t=t_cam2gripper
        )

        print("✅ Hand–Eye Calibration complete (Checkerboard).")
        print("Camera → Gripper Transform:\n", T_cam2gripper)


if __name__ == "__main__":
    calibrator = HandEyeCalibration()
    calibrator.motion_planner.arm.freeMoveRobot()
    calibrator.run_calibration(num_samples=10)
