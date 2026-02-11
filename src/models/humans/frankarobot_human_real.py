import time
import copy

from src.models.humans.base_human import BaseHuman
from src.utils.bullet_utils import *


class FrankarobotHumanReal(BaseHuman):
    def __init__(self, params, env, featurized_trajs=None, probs=None, scaling_coeffs=None, train_object_centers=None, test_object_centers=None, **kwargs):
        super().__init__()
        # Set environment and features in the environment.
        self.env = env
        self.features = params["features"]
        self.feat_scaling = params["feature_scaling"]
        self.theta = None

        # self.stand_z = p.getBasePositionAndOrientation(self.env.objectID["stand"])[0][2]
        # self.human_xy = p.getBasePositionAndOrientation(self.env.objectID["human"])[0][0:2]
        if featurized_trajs is not None:
            self.featurized_trajs = featurized_trajs
        if probs is not None:
            self.probs = probs
        if scaling_coeffs is not None:
            self.scaling_coeffs = scaling_coeffs
        if train_object_centers is not None:
            self.train_object_centers = train_object_centers
        if test_object_centers is not None:
            self.test_object_centers = test_object_centers

    def __str__(self):
        return "FrankarobotHumanReal " + str(self.theta)

    def calc_features(self, traj, object_centers=None):
        assert hasattr(self, 'features'), "Human does not have features yet."
        features = self.calc_features_cached(traj, object_centers=object_centers)
        if len(self.scaling_coeffs) != 0:
            for feat in range(len(features)):
                if self.feat_scaling == "normalize":
                    features[feat] = (features[feat] - self.scaling_coeffs[feat]["min"]) / (self.scaling_coeffs[feat]["max"] - self.scaling_coeffs[feat]["min"])
                elif self.feat_scaling == "standardize":
                    features[feat] = (features[feat] - self.scaling_coeffs[feat]["mu"]) / self.scaling_coeffs[feat]["sigma"]
        return np.array(features)

    def calc_features_cached(self, traj, object_centers=None):
        traj = np.array(traj)
        features = []
        for feat in self.features:
            if feat == "table":
                features.extend(self.table_features(traj, object_centers=object_centers))
            elif feat == "human":
                features.extend(self.human_features(traj, object_centers=object_centers))
            elif feat == "laptop":
                features.extend(self.laptop_features(traj, object_centers=object_centers))
            elif feat == "proxemics":
                features.extend(self.proxemics_features(traj, object_centers=object_centers))
            elif feat == "coffee":
                features.extend(self.coffee_features(traj, object_centers=object_centers))
            else:
                raise NotImplementedError
        return features
    
    def calc_eef_object_states(self, traj, object_centers=None, state_dim=9):
        """
        Returns a (T x 13) state matrix for the full trajectory, where T is the number of waypoints.
        Each state vector is constructed as follows:
        - EEF position: 3 dimensions (extracted from indices 100:103)
        - EEF rotation: 4 dimensions (extracted from indices 103:107)
        - Table height: 1 dimension (from the environment, assumed constant)
        - Human xy position: 2 dimensions (from the environment, assumed constant)
        - Laptop xy position: 2 dimensions (from the environment, assumed constant)
        
        Args:
            traj: list of waypoints (each waypoint is assumed to be a numpy array or list with sufficient dimensions)
            
        Returns:
            states: numpy array of shape (T, 9)
        """
        states = []
        
        for waypoint in traj:
            if state_dim == 9:
                # Extract end-effector position and rotation from the waypoint.
                # Adjust indices based on your trajectory data format.
                eef_pos = np.array(waypoint[:3])    # 3 DoF position
                eef_rot_quat = np.array(waypoint[3:7]) # 1 DoF rotation (z-axis)
                eef_rot_matrix = p.getMatrixFromQuaternion(eef_rot_quat)
                eef_rot = np.array([eef_rot_matrix[6]])  # Extract the z-axis component from the rotation matrix
                laptop_pos = np.array(waypoint[17:19])  # 2 DoF laptop 
                human_xy = np.array(waypoint[14:16])  # 2 DoF human position
                table_height = 0 # 1 DoF table height
                # Concatenate into a single state vector.
                state = np.concatenate((eef_pos, eef_rot, 
                                        np.array([table_height]), 
                                        np.array(human_xy), 
                                        np.array(laptop_pos)))
            elif state_dim == 19:
                eef_pos = np.array(waypoint[:3])    # 3 DoF position
                # use 9-dim rotation matrix 
                eef_rot_quat = np.array(waypoint[3:7]) # 1 DoF rotation (z-axis)
                eef_rot = p.getMatrixFromQuaternion(eef_rot_quat)
                laptop_pos = np.array(waypoint[17:20])  # 2 DoF laptop 
                human_xy = np.array(waypoint[14:17])  # 2 DoF human position
                table_height = 0  # 1 DoF table height
                # Concatenate into a single state vector.
                state = np.concatenate((eef_pos, eef_rot, 
                                        np.array(human_xy), 
                                        np.array(laptop_pos),
                                        np.array([table_height])))
            elif state_dim == 11:
                eef_pos = np.array(waypoint[:3])    # 3 DoF position
                # use 9-dim rotation matrix 
                eef_rot_quat = np.array(waypoint[3:7]) # 1 DoF rotation (z-axis)
                eef_rot = p.getMatrixFromQuaternion(eef_rot_quat)
                eef_rot = np.array([eef_rot[6]])
                laptop_pos = np.array(waypoint[17:19])  # 3 DoF laptop position
                human_xy = np.array(waypoint[14:16])  # 3 DoF human position
                table_height = 0  # 1 DoF table height
                # Concatenate into a single state vector.
                state = np.concatenate((eef_pos, eef_rot, 
                                        np.array(human_xy), 
                                        np.array(laptop_pos),
                                        np.array([table_height])))
            states.append(state)
        return np.array(states)

    # -- Distance to Table -- #

    def table_features(self, traj, object_centers=None):
        """
        Computes the total feature value over waypoints based on z-axis distance to table.
        ---
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar feature sum
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            if object_centers is None:
                stand_z = 0 #p.getBasePositionAndOrientation(self.env.objectID["stand"])[0][2] # support height
            else:
                stand_z = 0 #p.getBasePositionAndOrientation(self.env.objectID["stand"])[0][2] + object_centers["TABLE_CENTER"][2]
            feat_val += waypt[2] - stand_z
        return feat_val

    # -- Distance to Laptop -- #

    def laptop_features(self, traj, object_centers=None):
        """
        Computes distance from end-effector to laptop in xy coords.
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar distance sum where
                0: EE is at more than 0.3 meters away from laptop
                +: EE is closer than 0.3 meters to laptop
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            # if object_centers is None:
            dist = np.linalg.norm(waypt[:2] - waypt[17:19]) - 0.8
            # else:
            #     dist = np.linalg.norm(waypt[:2] - object_centers["LAPTOP_CENTER"][:2]) - 0.8
            feat_val += -((dist < 0) * dist)
        return feat_val

    # -- Distance to Human -- #

    def human_features(self, traj, object_centers=None):
        """
        Computes distance from end-effector to human in xy coords.
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar distance sum where
                0: EE is at more than 0.3 meters away from human
                +: EE is closer than 0.3 meters to human
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            EE_coord_xy = waypt[:2]
            # if object_centers is None:
            #     human_xy = list(self.human_xy)
            # else:
            #     human_xy = object_centers["HUMAN_CENTER"][:2]
            human_xy = waypt[14:16]
            dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.8
            feat_val += -((dist < 0) * dist)
        return feat_val

    # -- Proxemics -- #

    def proxemics_features(self, traj, object_centers=None):
        """
        Computes distance from end-effector to human proxemics in xy coords.
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar distance sum where
                0: EE is at more than 0.3 meters away from human
                +: EE is closer than 0.3 meters to human
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            EE_coord_xy = copy.deepcopy(waypt[:2])
            # if object_centers is None:
            #     human_xy = copy.deepcopy(list(self.human_xy))
            # else:
            #     human_xy = object_centers["HUMAN_CENTER"][:2]
            human_xy = copy.deepcopy(waypt[14:16])

            # Modify ellipsis distance.
            EE_coord_xy[1] /= 3
            human_xy[1] /= 3
            dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.8
            feat_val += -((dist < 0) * dist)
        return feat_val

    # -- Coffee -- #
    def coffee_features(self, traj, object_centers=None):
        """
        Computes the coffee orientation feature value as the EE orientation.
        ---
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar feature sum
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            EE_orient_quat = waypt[3:7]
            EE_orient_rot_matrix = p.getMatrixFromQuaternion(EE_orient_quat)
            EE_orient_x = EE_orient_rot_matrix[6]
            feat_val += (1 - EE_orient_x)
        return feat_val

    def test_features(self):
        for (idx, feat) in enumerate(self.features):
            steps = 500
            while steps:
                state_info = p.getJointStates(self.env.objectID["robot"], range(11))
                traj = [s[0] for s in state_info[1:]]
                if feat == "table":
                    feat_val = self.table_features(traj)
                elif feat == "human":
                    feat_val = self.human_features(traj)
                elif feat == "laptop":
                    feat_val = self.laptop_features(traj)
                elif feat == "proxemics":
                    feat_val = self.proxemics_features(traj)
                elif feat == "coffee":
                    feat_val = self.coffee_features(traj)
                else:
                    raise NotImplementedError
                print("Feature {} value: {}".format(feat, feat_val))
                steps -= 1
                time.sleep(0.01)
