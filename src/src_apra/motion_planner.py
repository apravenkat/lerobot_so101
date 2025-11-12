import os
import time
import mujoco
import mujoco.viewer
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig
from ompl import base as ob
from ompl import geometric as og
import numpy as np
np.float = float
from urdfpy import URDF


class MotionPlanner:
    def __init__(self, xml_file="so101_new_calib.xml", urdf_file="so101_new_calib.urdf"):
        self.HERE = os.path.dirname(__file__)
        self.xml_path = os.path.join(self.HERE, xml_file)
        self.urdf_path = os.path.join(self.HERE, urdf_file)
        self.model = self.load_model()
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.joint_name_to_index = {self.model.joint(i).name: i for i in range(self.model.njnt)}
        self.last_sent_pose = None
        self.joint_indices = [self.model.joint(name).id for name in self.joint_names]
        self.smoothing_alpha = 0.8  # Smoothing factor for joint movements

        robot_urdf = URDF.load(self.urdf_path)
        self.joint_limits = {}
        for joint in robot_urdf.joints:
            if joint.limit is not None:
                self.joint_limits[joint.name] = (joint.limit.lower, joint.limit.upper)

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
    
    def get_current_pos_end(self):
        """Get current end-effector position."""
        mujoco.mj_forward(self.model, self.data)
        end_effector_site_id = self.model.site("gripperframe").id
        pos = self.data.site_xpos[end_effector_site_id]
        return pos.copy()

    def visualize(self, timestep=0.01):
        """Launch viewer and run simulation loop."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                current_pos = planner.get_current_pos_end()
                print(f"Current End-Effector Position: {current_pos}")
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    def  inverse_kinematics(self, target_pos, target_rot=None, sit_name="gripperframe", max_iter=200, tol=1e-3, step_size=0.05):
        #site_id = self.model.site(sit_name).id
        ik_converged = False
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sit_name)
        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[site_id]
            current_rot = self.data.site_xmat[site_id].reshape(3, 3)    
            pos_error = target_pos - current_pos
            if target_rot is not None:
                R_err = target_rot @ current_rot.T
                rot_err = 0.5 * (np.array([R_err[2, 1] - R_err[1, 2],
                                           R_err[0, 2] - R_err[2, 0],
                                           R_err[1, 0] - R_err[0, 1]]))
                error6 = np.concatenate((pos_error, rot_err))
            else:
                error6 = pos_error
            if np.linalg.norm(error6) < tol:
                print("IK converged")
                ik_converged = True
                break
            J_pos = np.ascontiguousarray(np.zeros((3, self.model.nv)), dtype=np.float64)
            J_rot = np.ascontiguousarray(np.zeros((3, self.model.nv)), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, J_pos, None, site_id)
            if target_rot is not None:
                J = np.vstack((J_pos, J_rot))
            else:
                J = J_pos
            lambda_ = 0.001
            JJt = J @ J.T + lambda_ * np.eye(J.shape[0])
            dq = step_size * J.T @ np.linalg.solve(JJt, error6)
            self.data.qpos[:self.model.nq] += dq
        mujoco.mj_forward(self.model, self.data)
        print("Final end-effector position:", self.data.site_xpos[site_id])
        return self.data.qpos[:self.model.nv].copy(), ik_converged
    
    def get_base2gripper_transform(self):
        
        joint_locations = self.getRobotPose()
        self.data.qpos[:len(self.joint_names)] = np.radians(joint_locations)
        mujoco.mj_forward(self.model, self.data)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        pos = self.data.site_xpos[site_id]
        rot = self.data.site_xmat[site_id].reshape(3, 3)
        T_base2gripper = np.eye(4)
        T_base2gripper[:3, :3] = rot
        T_base2gripper[:3, 3] = pos
        return T_base2gripper

    def plan_motion(self, target_pos, target_rot, site_name="gripperframe", plan_time=2.0, interpolate_points=50):
        start_q = self.data.qpos.copy()
        goal_q, ik_convereged = self.inverse_kinematics(target_pos, target_rot, sit_name=site_name)
        if not ik_convereged:
            print("IK did not converge, cannot plan motion.")
            return None
        print("Goal configuration ",goal_q)
        dim = len(start_q)
        space = ob.RealVectorStateSpace(dim)
        
        bounds = ob.RealVectorBounds(dim)
        for i, name in enumerate(self.joint_names):
            lower, upper = self.joint_limits.get(name, (-np.pi, np.pi))
            bounds.setLow(i, lower)
            bounds.setHigh(i, upper)
        space.setBounds(bounds)
        
        def is_valid(state):
            q = np.array([state[i] for i in range(dim)])
            self.data.qpos[:dim] = q
            mujoco.mj_forward(self.model, self.data)
            return True
        
        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
        si.setup()
        start = ob.State(space)
        goal = ob.State(space)
        for i in range(dim):
            start[i] = start_q[i]
            goal[i] = goal_q[i]
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        planner = og.RRTstar(si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(plan_time)
        if not solved:
            print("No solution found")
            return None
        path = pdef.getSolutionPath()
        path.interpolate(interpolate_points)
        traj = np.array([[s[i] for i in range(dim)] for s in path.getStates()])
        return traj
    
    def plan_motion_to_joint_positions(self, target_q, plan_time=2.0, interpolate_points=50):
        start_q = self.data.qpos.copy()
        goal_q = target_q
        dim = len(start_q)
        space = ob.RealVectorStateSpace(dim)
        
        bounds = ob.RealVectorBounds(dim)
        for i, name in enumerate(self.joint_names):
            lower, upper = self.joint_limits.get(name, (-np.pi, np.pi))
            bounds.setLow(i, lower)
            bounds.setHigh(i, upper)
        space.setBounds(bounds)
        for i, name in enumerate(self.joint_names):
            low, high = self.joint_limits.get(name, (-np.pi, np.pi))
            print(f"{name}: {start_q[i]:.5f} vs bounds ({low:.5f}, {high:.5f})")
        def is_valid(state):
            q = np.array([state[i] for i in range(dim)])
            self.data.qpos[:dim] = q
            mujoco.mj_forward(self.model, self.data)
            return True
        
        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
        si.setup()
        start = ob.State(space)
        goal = ob.State(space)
        for i in range(dim):
            start[i] = start_q[i]
            goal[i] = goal_q[i]
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        planner = og.RRTstar(si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(plan_time)
        if not solved:
            print("No solution found")
            return None
        path = pdef.getSolutionPath()
        path.interpolate(interpolate_points)
        traj = np.array([[s[i] for i in range(dim)] for s in path.getStates()])
        return traj
    
    def visualize_trajectory(self, traj, timestep=0.05):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for qpos in traj:
                smoothed_qpos = self.smooth_joint_positions((qpos[:len(self.joint_names)]))
                self.data.qpos[:len(qpos)] = smoothed_qpos
                mujoco.mj_forward(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(timestep)

    def getRobotPose(self):
        """Get current robot joint positions."""
        observation = self.arm.get_observation()
        joint_positions = []
        for name, angle in observation.items():
            if name.endswith(".pos"):
                joint_positions.append(angle)
        return joint_positions
    
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
    def sendSimPoseToRobot(self, joint_pos):
        action = {name + ".pos": float(angle) for name, angle in zip(self.joint_names, joint_pos)}
        try:
            sent_action = self.arm.send_action(action)
        except Exception as e:
            print(f"Failed to send joint angles to robot: {e}")
        time.sleep(0.01)
        
    def getEndEffectorPosition(self):
        mujoco.mj_forward(self.model, self.data)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        pos = self.data.site_xpos[site_id]
        print(f"End-Effector Position: {pos}")
        return pos.copy()

if __name__ == "__main__":
    planner = MotionPlanner()
    sim = False
    planner.arm.freeMoveRobot()
    current_robot_joint_positions = planner.getRobotPose()
    print("Current Robot Joint Positions (radians):", np.radians(current_robot_joint_positions))
    planner.data.qpos[:len(planner.joint_names)] = np.radians(current_robot_joint_positions)
    pos = planner.getEndEffectorPosition()
    print(f"Current End-Effector Position: {pos}")
    '''
    if sim:
        current_position = planner.get_current_pos_end()
        print(f"Current End-Effector Position: {current_position}")
        desired_position = current_position + np.array([0.0, 0.0, 0.1])
        print(f"Planning motion to {desired_position}")
        traj = planner.plan_motion(desired_position)
        if traj is None:
            print("Failed to find a motion plan.")
        else:
            print(traj)
            print("Motion plan completed.")
            planner.visualize_trajectory(traj, timestep=0.05)
            print("Visualization completed.")
    else:
        current_robot_joint_positions = planner.getRobotPose()
        planner.data.qpos[:len(planner.joint_names)] = np.radians(current_robot_joint_positions)
        site_id = mujoco.mj_name2id(planner.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        mujoco.mj_forward(planner.model, planner.data)
        current_position = planner.data.site_xpos[site_id]
        print(f"Current End-Effector Position: {current_position}")
        desired_position = current_position + np.array([0.1, 0.0, 0.1])
        print(f"Planning motion to {desired_position}")
        traj = planner.plan_motion(desired_position)
        if traj is None:
            print("Failed to find a motion plan.")
        else:
            print(traj)
            print("Motion plan completed.")
            planner.visualize_trajectory(traj, timestep=0.05)
            print("Visualization completed.")
            print("Do you want to execute this motion on the real robot? (y/n)")
            choice = input().strip().lower()
            if choice != 'y':
                print("Motion execution cancelled.")
                exit(0)
            for qpos in traj:
                smoothed_qpos = planner.smooth_joint_positions(np.degrees(qpos[:len(planner.joint_names)]))
                #planner.data.qpos[:len(planner.joint_names)] = np.radians(smoothed_qpos)
                #mujoco.mj_forward(planner.model, planner.data)
                planner.sendSimPoseToRobot(smoothed_qpos)
            print("Motion execution completed.") '''