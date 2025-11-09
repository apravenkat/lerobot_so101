import os
import time
import mujoco
import mujoco.viewer
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig
import numpy as np

class GripperSimulator:
    def __init__(self, xml_file="so101_new_calib.xml", urdf_file="so101_new_calib.urdf", gripper_joint_name="gripper"):
        self.HERE = os.path.dirname(__file__)
        self.xml_path = os.path.join(self.HERE, xml_file)
        self.urdf_path = os.path.join(self.HERE, urdf_file)
        self.model = self.load_model()
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.joint_name_to_index = {self.model.joint(i).name: i for i in range(self.model.njnt)}
        self.last_sent_pose = None
        self.smoothing_alpha = 0.8  # Smoothing factor for joint movements
        # Gripper parameters
        self.min_open = 0.0
        self.max_open = 1.75
        self.speed = 1.0
        self.opening = True
        self.config = SO101FollowerConfig(port="/dev/ttyACM0", use_degrees=True)
        self.arm = SO101Follower(self.config)
        self.arm.connect(calibrate=False)
        

    def load_model(self):
        """Load XML model first, fallback to URDF if XML fails."""
        try:
            model = mujoco.MjModel.from_xml_path(self.xml_path)
            print(f"✅ Loaded model from {self.xml_path}")
        except Exception as e:
            print(f"⚠️ Failed to load XML: {e}")
            print(f"➡️ Trying URDF instead...")
            model = mujoco.MjModel.from_xml_path(self.urdf_path)
            print(f"✅ Loaded model from {self.urdf_path}")
        return model
    
    def getRobotPose(self):
        """Get current robot joint positions."""
        observation = self.arm.get_observation()
        joint_positions = []
        for name, angle in observation.items():
            if name.endswith(".pos"):
                joint_positions.append(angle)
        return joint_positions
    def run(self, timestep=0.1):
        """Launch viewer and run simulation loop."""
        count = 0
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                joint_pos = self.getRobotPose()
                '''for i in range(len(joint_pos)):
                    self.data.qpos[i] = joint_pos[i]*np.pi/180'''
                print(np.radians(joint_pos))
                self.data.qpos[:len(joint_pos)] = np.radians(joint_pos)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(timestep)
    
    def smooth_joint_positions(self, target_deg):
        """
        Apply exponential smoothing to avoid sudden jumps.
        new = α * target + (1 - α) * previous
        """
        if self.last_sent_pose is None:
            self.last_sent_pose = target_deg.copy()
        else:
            self.last_sent_pose = (
                self.smoothing_alpha * np.array(target_deg)
                + (1 - self.smoothing_alpha) * np.array(self.last_sent_pose)
            )
        return self.last_sent_pose.tolist()
    
    def sendSimPoseToRobot(self):
        """
        Send simulated joint angles (radians) to the real robot with smoothing.
        First call is safely initialized from the robot's current joint positions.
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                qpos_deg = np.degrees(self.data.qpos[:len(self.joint_names)])

                if self.last_sent_pose is None:
                    try:
                        current_robot_pose = self.getRobotPose()
                        print("ℹ️ Initializing smoothing from robot’s current pose:", current_robot_pose)
                        self.last_sent_pose = current_robot_pose
                    except Exception as e:
                        print(f"⚠️ Could not read robot pose for initialization: {e}")
                

                # Apply smoothing
                smoothed_deg = self.smooth_joint_positions(qpos_deg[4])
                #smoothed_deg = np.clip(smoothed_deg, 0, 100)

                #action = {name + ".pos": float(angle) for name, angle in zip(self.joint_names[5], smoothed_deg)}
                joint_index = 4  # target joint index (0-based)
                joint_name = self.joint_names[joint_index]
                joint_angle = float(smoothed_deg[joint_index])
                observation = self.getRobotPose()
                action = {joint_name + ".pos": joint_angle}

                
                print(action)
                try:
                    sent_action = self.arm.send_action(action)
                    print("➡️ Sent smoothed joint angles to robot:", sent_action)
                except Exception as e:
                    print(f"⚠️ Failed to send joint angles to robot: {e}")
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    sim = GripperSimulator()
    sim.arm.freeMoveRobot()
    sim.run(timestep=0.05)
