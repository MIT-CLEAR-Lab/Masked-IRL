import copy
import math

import plotly.graph_objects as go
from scipy import special
import pybullet as p

from src.utils.bullet_utils import *
from src.utils.trajectory import Trajectory
from pybullet_utils import bullet_client
import pybullet

class Frankarobot(object):

    def __init__(self, object_centers, resources_dir, horizon, timestep, debug=False):
        # Start GUI
        self.debug = debug
        self.object_centers = object_centers
        self.objectID = start_environment(object_centers, resources_dir, direct=(not self.debug), robot_type="franka")
        self.horizon = horizon
        self.timestep = timestep
        self.robot_type = "franka"

    # def replay(self, trajs, colors):
    #     while True:
    #         for traj, color in zip(trajs, colors):
    #             replay_trajectory(self.objectID, traj, color=color)
    #         time.sleep(0.01)
    def replay(self, trajs, colors, forever=True, show_ee_coords=False):
        while True:
            for traj, color in zip(trajs, colors):
                replay_trajectory(self.objectID, traj, color=color, show_ee_coords=show_ee_coords)
            if not forever:
                break
            time.sleep(0.01)

    def query_pref(self, sample1, sample2, gt=None):
        assert self.debug, "GUI not active."

        # Add relevant buttons.
        p.addUserDebugParameter(str(gt), 1, 0, 0)
        replay1_button = p.addUserDebugParameter("Replay Traj 1", 1, 0, 0)
        replay2_button = p.addUserDebugParameter("Replay Traj 2", 1, 0, 0)
        pref1_button = p.addUserDebugParameter("Red better than Blue", 1, 0, 0)
        pref2_button = p.addUserDebugParameter("Blue better than Red", 1, 0, 0)
        top1_button = p.addUserDebugParameter("View #1", 1, 0, 0)
        top2_button = p.addUserDebugParameter("View #2", 1, 0, 0)
        side_button = p.addUserDebugParameter("View #3", 1, 0, 0)
        replay1_num = 0
        replay2_num = 0
        top1_num = 0
        top2_num = 0
        side_num = 0

        # Play query one by one.
        replay_trajectory(self.objectID, sample1, color=[1, 0, 0])
        replay_trajectory(self.objectID, sample2, color=[0, 0, 1])

        while True:
            # Operate buttons.
            replay1_pushes = p.readUserDebugParameter(replay1_button)
            replay2_pushes = p.readUserDebugParameter(replay2_button)
            pref1_pushes = p.readUserDebugParameter(pref1_button)
            pref2_pushes = p.readUserDebugParameter(pref2_button)
            top1_pushes = p.readUserDebugParameter(top1_button)
            top2_pushes = p.readUserDebugParameter(top2_button)
            side_pushes = p.readUserDebugParameter(side_button)

            if top1_pushes > top1_num:
                top1_num = top1_pushes
                xview()

            if top2_pushes > top2_num:
                top2_num = top2_pushes
                yview()

            if side_pushes > side_num:
                side_num = side_pushes
                zview()

            if replay1_pushes > replay1_num:
                replay1_num = replay1_pushes
                replay_trajectory(self.objectID, sample1, color=[1, 0, 0])

            if replay2_pushes > replay2_num:
                replay2_num = replay2_pushes
                replay_trajectory(self.objectID, sample2, color=[0, 0, 1])

            if pref1_pushes > 0:
                p.removeAllUserParameters()
                return 0

            if pref2_pushes > 0:
                p.removeAllUserParameters()
                return 1

            time.sleep(0.01)

    def query_triplet(self, anchor, sample1, sample2, gt=None):
        assert self.debug, "GUI not active."

        # Add relevant buttons.
        p.addUserDebugParameter(str(gt), 1, 0, 0)
        replay1_button = p.addUserDebugParameter("Replay Traj 1", 1, 0, 0)
        replay2_button = p.addUserDebugParameter("Replay Traj 2", 1, 0, 0)
        replay3_button = p.addUserDebugParameter("Replay Traj 3", 1, 0, 0)
        sim1_button = p.addUserDebugParameter("Red more similar to Green", 1, 0, 0)
        sim2_button = p.addUserDebugParameter("Red more similar to Blue", 1, 0, 0)
        sim3_button = p.addUserDebugParameter("Green more similar to Blue", 1, 0, 0)
        top1_button = p.addUserDebugParameter("View #1", 1, 0, 0)
        top2_button = p.addUserDebugParameter("View #2", 1, 0, 0)
        side_button = p.addUserDebugParameter("View #3", 1, 0, 0)
        replay1_num = 0
        replay2_num = 0
        replay3_num = 0
        top1_num = 0
        top2_num = 0
        side_num = 0

        # Play the query one by one.
        replay_trajectory(self.objectID, anchor, color=[1, 0, 0])
        replay_trajectory(self.objectID, sample1, color=[0, 1, 0])
        replay_trajectory(self.objectID, sample2, color=[0, 0, 1])

        while True:
            # Operate buttons.
            replay1_pushes = p.readUserDebugParameter(replay1_button)
            replay2_pushes = p.readUserDebugParameter(replay2_button)
            replay3_pushes = p.readUserDebugParameter(replay3_button)
            sim1_pushes = p.readUserDebugParameter(sim1_button)
            sim2_pushes = p.readUserDebugParameter(sim2_button)
            sim3_pushes = p.readUserDebugParameter(sim3_button)
            top1_pushes = p.readUserDebugParameter(top1_button)
            top2_pushes = p.readUserDebugParameter(top2_button)
            side_pushes = p.readUserDebugParameter(side_button)

            if top1_pushes > top1_num:
                top1_num = top1_pushes
                xview()

            if top2_pushes > top2_num:
                top2_num = top2_pushes
                yview()

            if side_pushes > side_num:
                side_num = side_pushes
                zview()

            if replay1_pushes > replay1_num:
                replay1_num = replay1_pushes
                replay_trajectory(self.objectID, anchor, color=[1, 0, 0])

            if replay2_pushes > replay2_num:
                replay2_num = replay2_pushes
                replay_trajectory(self.objectID, sample1, color=[0, 1, 0])

            if replay3_pushes > replay3_num:
                replay3_num = replay3_pushes
                replay_trajectory(self.objectID, sample2, color=[0, 0, 1])

            if sim1_pushes > 0:
                p.removeAllUserParameters()
                p.removeAllUserDebugItems()
                return anchor, sample1, sample2

            if sim2_pushes > 0:
                p.removeAllUserParameters()
                p.removeAllUserDebugItems()
                return anchor, sample2, sample1

            if sim3_pushes > 0:
                p.removeAllUserParameters()
                p.removeAllUserDebugItems()
                return sample1, sample2, anchor

            view_matrix = p.computeViewMatrix(cameraEyePosition=[0, 0, 3], cameraTargetPosition=[0, 0, 0], cameraUpVector=[0, 1, 0])
            projection_matrix = p.computeProjectionMatrixFOV(fov=45.0,
                                                             aspect=1.0,
                                                             nearVal=0.1,
                                                             farVal=10)

            # Get depth values using the OpenGL renderer
            p.getCameraImage(256, 256, view_matrix, projection_matrix, shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            time.sleep(0.01)

    def generate_trajs(self, samples, per_SG=10):
        """
        Generates a set of trajectories.

        Params:
            samples [int] -- Number of SG trajectory samples.
            per_SG [int] -- Number of samples per SG pair.

        Returns:
            trajectory -- Deformed trajectory.
        """
        # Initialize buttons if debug mode.
        if self.debug:
            next_button = p.addUserDebugParameter("Next Sample", 1, 0, 0)
            next_num = 0

        # Collect data.
        sampling = True
        trajs = []
        while len(trajs) < samples*per_SG:
            # Operate buttons if debugging.
            if self.debug:
                next_pushes = p.readUserDebugParameter(next_button)
                if next_pushes > next_num:
                    next_num = next_pushes
                    if sampling == False:
                        p.removeAllUserDebugItems()
                        sampling = True

            # Sample a trajectory via deformation.
            if sampling:
                # Sample random S and G poses.
                #move_laptop(self.objectID["laptop"])
                path_length = int(self.horizon / self.timestep) + 1
                start_pos, goal_pos = random_SG_pos(self.objectID, path_length)

                # Generate multiple trajectories per SG pair.
                samples_SG = 0
                trajs_SGs = []
                while samples_SG < per_SG:
                    # Get trajectory from start to goal.
                    start_pose = random_legalpose(self.objectID["robot"], pos=start_pos)
                    goal_pose = random_legalpose(self.objectID["robot"], pos=goal_pos)

                    # Compute straight line path in configuration space.
                    waypts = np.linspace(start_pose, goal_pose, path_length)
                    waypts_time = np.linspace(0.0, self.horizon, path_length)
                    traj = Trajectory(waypts, waypts_time, robot_type=self.robot_type)

                    # Visualize base trajectory.
                    if self.debug:
                        plot_trajectory(waypts_to_xyz(self.objectID["robot"], waypts), color=[0, 1, 0])

                    # Perturb trajectory such that it is different enough from initial.
                    traj_delta = 0
                    while traj_delta < 8.0:
                        deformed_traj = copy.deepcopy(traj)
                        # Sample number of waypoints to perturb.
                        num_waypts_deform = random.randint(1, 3)
                        # Perturb trajectory.
                        for _ in range(num_waypts_deform):
                            # Choose deformation magnitude and width.
                            alpha = np.random.uniform(-0.1, 0.1)
                            n = random.randint(8, path_length - 2)

                            # Sample perturbation vector and waypoint to apply it to.
                            u = np.random.uniform(low=-math.pi, high=math.pi, size=7)
                            u = np.hstack((u, [0, 0, 0])).reshape((-1, 1))
                            waypt_idx = random.randint(1, path_length - n)

                            # Deform trajectory.
                            deformed_traj = deformed_traj.deform(u, waypt_idx * self.timestep, alpha, n)
                            print("Deformed trajectory at {} waypt, {} alpha, {} n".format(waypt_idx, alpha, n))
                        # Validate deformed trajectory.
                        traj_delta = sum(np.linalg.norm(deformed_traj.waypts - traj.waypts, axis=1))
                        traj_delta = min([traj_delta] + [sum(np.linalg.norm(traj_sg.waypts - deformed_traj.waypts, axis=1)) for traj_sg in trajs_SGs])

                    # Visualize sampled trajectory if in debug mode.
                    if self.debug:
                        # View trajectory.
                        plot_trajectory(waypts_to_xyz(self.objectID["robot"], deformed_traj.waypts), color=[0, 0, 1])
                        sampling = False
                    # Save the trajectory.
                    samples_SG += 1
                    trajs.append(list(waypts_to_raw(self.objectID, deformed_traj.waypts)))
                    trajs_SGs.append(copy.deepcopy(deformed_traj))
                    print("--------------------- {} / {} -----------------------".format(len(trajs), samples*per_SG))

            time.sleep(0.01)
        if self.debug:
            visualize_trajset(self.objectID, trajs)
            p.removeAllUserParameters()
        return trajs

    def show_EE_coords(self):
        #get current state of EE
        state = p.getLinkState(self.objectID['robot'], 8)
        p_WE_W = np.array(state[0])
        R_WE = np.array(p.getMatrixFromQuaternion(state[1])).reshape(3,3)
        #Get positions of each x,y,z EE axis in world frame
        p_WX_W = np.matmul(R_WE, np.array([1,0,0])) + p_WE_W
        p_WY_W = np.matmul(R_WE, np.array([0,1,0])) + p_WE_W
        p_WZ_W = np.matmul(R_WE, np.array([0,0,1])) + p_WE_W
        #display axes
        p.addUserDebugLine(p_WE_W, p_WX_W, lineColorRGB=[1,0,0], lineWidth=3.0)
        p.addUserDebugLine(p_WE_W, p_WY_W, lineColorRGB=[0,1,0], lineWidth=3.0)
        p.addUserDebugLine(p_WE_W, p_WZ_W, lineColorRGB=[0,0,1], lineWidth=3.0)
    

    def visualize(self, trajs, rews, is_probs=False, im_path=None, max_display=100, custom_colors=None, fig=None):
        # Plot human and laptop if they exist.
        posH, _ = p.getBasePositionAndOrientation(self.objectID["human"])
        posL, _ = p.getBasePositionAndOrientation(self.objectID["laptop"])
        if fig is None:
            fig = go.Figure(data=[go.Scatter3d(x=(posH[0], posL[0]), y=(posH[1], posL[1]), z=(posH[2], posL[2]), mode='markers',
                                            marker=dict(color=("gray", "black"), symbol=("cross", "square"), showscale=False), showlegend=False)])

        # Plot top trajectories in the color of their rewards.
        probs = rews / np.sum(rews) if is_probs else special.softmax(np.array(rews))
        order = np.argsort(-probs)
        for idx, traj in enumerate(trajs[order]):
            if idx == max_display:
                break
            xyz_traj = waypts_to_xyz(self.objectID["robot"], traj)
            if max(probs) - min(probs) != 0:
                traj_opacity = (probs[idx] - min(probs)) / (max(probs) - min(probs))
            else:
                traj_opacity = 1
            # if traj_opacity is nan, set it to 1.
            if math.isnan(traj_opacity):
                traj_opacity = 1
            if custom_colors is None:
                fig.add_trace(go.Scatter3d(x=xyz_traj[:, 0], y=xyz_traj[:, 1], z=xyz_traj[:, 2], mode='lines',
                                        opacity=traj_opacity, line=dict(color='red', width=10), showlegend=False))
            else:
                fig.add_trace(go.Scatter3d(x=xyz_traj[:, 0], y=xyz_traj[:, 1], z=xyz_traj[:, 2], mode='lines',
                                        opacity=traj_opacity, line=dict(color=custom_colors[idx], width=10), showlegend=False))

        if im_path:
            im_path = im_path[:-3] + "html"
            fig.write_html(im_path)
        else:
            fig.show()

        return fig

    def visualize_embedding(self, trajs, featurizer, num_encoders, dims):
        posH, _ = p.getBasePositionAndOrientation(self.objectID["human"])
        posL, _ = p.getBasePositionAndOrientation(self.objectID["laptop"])

        while True:
            selection = int(input("Selection: "))
            # Plot human and laptop if they exist.
            fig = go.Figure(data=[go.Scatter3d(x=(posH[0], posL[0]), y=(posH[1], posL[1]), z=(posH[2], posL[2]), mode='markers',
                                               marker=dict(color=("gray", "black"), symbol=("cross", "square"), showscale=False), showlegend=False)])
            # Plot selected trajectory.
            xyz_traj = waypts_to_xyz(self.objectID["robot"], trajs[selection])
            fig.add_trace(go.Scatter3d(x=xyz_traj[:, 0], y=xyz_traj[:, 1], z=xyz_traj[:, 2], mode='lines',
                                       line=dict(color='black', width=10), showlegend=False))

            # Select and plot best 10.
            featurized_trajs = featurizer.featurize(trajs).detach().cpu().numpy()
            distances = np.linalg.norm(featurized_trajs[selection][np.newaxis, :] - featurized_trajs, axis=1)
            indices = np.argsort(distances)[1:10]
            probs = special.softmax(-distances[indices])
            selected_trajs = np.array(trajs)[indices]

            # Plot top trajectories in the color of their distances.
            for traj, prob in zip(selected_trajs, probs):
                xyz_traj = waypts_to_xyz(self.objectID["robot"], traj)
                traj_opacity = (prob - min(probs)) / (max(probs) - min(probs))
                fig.add_trace(go.Scatter3d(x=xyz_traj[:, 0], y=xyz_traj[:, 1], z=xyz_traj[:, 2], mode='lines',
                                           opacity=traj_opacity, line=dict(color='red', width=10), showlegend=False))
            fig.show()

    def kill_env(self):
        # Disconnect once the session is over.
        p.disconnect()
