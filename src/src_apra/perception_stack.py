import cv2
import numpy as np
import motion_planner as planner
import glob
import os
import mujoco
import mujoco.viewer
import time
from motion_planner import MotionPlanner
import threading
import zmq
import pickle
import struct

class PerceptionStack:
    def __init__(self, device="/dev/video2", fps=30):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        # Force MJPEG (what USB cams actually support)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # Safe USB FPS
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Drop frames aggressively
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Reduce MJPEG decode issues
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.running = True
        self.fps = fps

        self.thread = threading.Thread(
            target=self._update_frame,
            daemon=True
        )
        self.thread.start()

    def _update_frame(self):
        target_interval = 1.0 / self.fps

        while self.running:
            start = time.time()

            # SAFE MJPEG PATH
            if not self.cap.grab():
                continue

            try:
                ret, frame = self.cap.retrieve()
            except cv2.error:
                continue

            if not ret or frame is None:
                continue

            with self.frame_lock:
                self.latest_frame = frame

            # Throttle to avoid USB saturation
            elapsed = time.time() - start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    

    def interinsinc_calibration(self):
        ChessBoardSize = (8, 5)
        SquareSize = 25
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objp = np.zeros((ChessBoardSize[0]*ChessBoardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:ChessBoardSize[0], 0:ChessBoardSize[1]].T.reshape(-1, 2)
        objp *= SquareSize
        objpoints = []
        imgpoints = []
        images = glob.glob('calibration_images/*.jpg')
        print(glob.glob('calibration_images/*.jpg'))
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, ChessBoardSize, None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, ChessBoardSize, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()
        ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez('camera_calibration_data.npz', camera_matrix=cam_mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)

    def get_pictures_for_calibration(self, num_images=15):
    
        count = 0
        os.makedirs('calibration_images', exist_ok=True)
        while count < num_images:
            frame = self.get_frame()
            if frame is None or frame.size == 0:
                print("No frame captured.")
                continue

            cv2.imshow('Calibration Frame', frame)
            key = cv2.waitKey(100) & 0xFF  # allow time for key input

            if key == ord('c'):
                filename = f'calibration_images/calib_{count}.jpg'
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"✅ Saved {filename}")
                    count += 1
                else:
                    print(f"❌ Failed to save {filename}")

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_image_coordinates_from_camera(self, camera_matrix, dist_coeffs, r_vecs, t_vecs):
        while(cv2.waitKey(1) & 0xFF != ord('q')):

            frame = self.get_frame()
            if frame is None or frame.size == 0:
                print("No frame captured.")
                return None
        
            
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(frame)
            marker_length = 0.036
            if ids is not None:
                
                for i in range(len(ids)):
                    img_points = corners[i][0]
                    half_len = marker_length / 2
                    obj_points = np.array([[-half_len, half_len, 0],
                                           [half_len, half_len, 0],
                                           [half_len, -half_len, 0],
                                           [-half_len, -half_len, 0]], dtype=np.float32)
                    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                    if success:
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                        print(tvec.ravel()) 
            cv2.imshow('Camera Frame', frame)
if __name__ == "__main__":
    perception = PerceptionStack(fps=30)

    while True:
        frame = perception.get_frame()
        if frame is not None:
            cv2.imshow("Camera Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    perception.stop()
    cv2.destroyAllWindows()

