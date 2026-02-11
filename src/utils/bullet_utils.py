import random
import time

import numpy as np
import pybullet as p

ROBOT_TYPE = None


# -- Functions for building the pybullet environment -- #

def start_environment(object_centers, resources_dir, direct=False, robot_type="franka"):
    # Connect to physics simulator.
    if direct:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)

    # Add path to data resources for the environment.
    p.setAdditionalSearchPath(resources_dir)

    # Setup the environment.
    global ROBOT_TYPE
    ROBOT_TYPE = robot_type
    objectID = setup_environment(object_centers)

    # Get rid of gravity and make simulation happen in real time.
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)

    return objectID


def setup_environment(object_centers):
    objectID = {}

    # Add the floor.
    objectID["plane"] = p.loadURDF("plane.urdf")

    # Add a table.
    pos = [-0.65, 0.0, 0.0]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["table"] = p.loadURDF("table/table.urdf", pos, orientation, useFixedBase=True)

    # Add a robot support.
    pos = [0.0, 0, 0.65]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    objectID["stand"] = p.loadURDF("support.urdf", pos, orientation, useFixedBase=True)

    # Add the laptop.
    if "LAPTOP_CENTER" in object_centers:
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        objectID["laptop"] = p.loadURDF("laptop.urdf", object_centers["LAPTOP_CENTER"], orientation, useFixedBase=True)

    # Add the human.
    if "HUMAN_CENTER" in object_centers:
        orientation = p.getQuaternionFromEuler([1.5, 0, 1.5])
        objectID["human"] = p.loadURDF("humanoid/humanoid.urdf", object_centers["HUMAN_CENTER"], orientation, globalScaling=0.25, useFixedBase=True)

    # Add the Jaco robot and initialize it.
    pos = [0.0, 0, 0.675]
    orientation = p.getQuaternionFromEuler([0, 0, 0])
    if ROBOT_TYPE == "franka":
        objectID["robot"] = p.loadURDF("franka_panda/panda.urdf", pos, orientation, useFixedBase=True)
    elif ROBOT_TYPE == "jaco":
        objectID["robot"] = p.loadURDF("jaco.urdf", pos, orientation, useFixedBase=True)
    move_robot(objectID["robot"])

    return objectID

def eef_pos(robotID):
    if ROBOT_TYPE == "jaco":
        return p.getLinkState(robotID, 7)[0]
    elif ROBOT_TYPE == "franka":
        return p.getLinkState(robotID, 8)[0]

# -- Functions for various kinematic robot operations -- #
def random_legal3Dpos(robotID):
    """
    Randomly choose a legal position above the table.
    """
    pos, _ = p.getBasePositionAndOrientation(robotID)
    limits = [(pos[0] - 0.6 - 1.0 / 2, pos[0] - 0.6 + 1.0 / 2), (pos[1] - 1.0 / 2, pos[1] + 1.0 / 2), (pos[2] + 0.05, pos[2] + 1.0)]
    target = [random.uniform(l[0], l[1]) for l in limits]
    return target


def random_legalpose(robotID, pos=None):
    """
    Randomly choose a legal position above the table.
    """
    if pos is None:
        pos = random_legal3Dpos(robotID)
    quat = np.random.randn(1, 4)
    quat = quat / np.linalg.norm(quat, axis=1)
    if ROBOT_TYPE == "jaco":
        joint_poses = p.calculateInverseKinematics(robotID, 7, pos, targetOrientation=quat[0], maxNumIterations=100)
    elif ROBOT_TYPE == "franka":
        joint_poses = p.calculateInverseKinematics(robotID, 8, pos, targetOrientation=quat[0], maxNumIterations=100)
    return joint_poses


def random_SG_pos(objectID, path_length, min_dist=0.7):
    """
    Sample a random pair of start and goal robot 3D EE positions.
    The sampled pair has to be min_dist away in cartesian space.
    """
    while True:
        # Sample random S and G positions.
        start_pos = random_legal3Dpos(objectID["robot"])
        goal_pos = random_legal3Dpos(objectID["robot"])
        distance = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
        if distance < min_dist:
            continue
        # Check if reasonably close to the laptop to be interesting.
        xyz_waypts = np.linspace(start_pos, goal_pos, path_length)
        posL, _ = p.getBasePositionAndOrientation(objectID["laptop"])
        dist_L = sum([0.8 - np.linalg.norm(np.array(waypt)[:2] - posL[:2]) for waypt in xyz_waypts])
        if dist_L < 1 and np.random.random() < 0.95:
            continue
        # Check if IK can solve both S and G.
        start_pose = random_legalpose(objectID["robot"], pos=start_pos)
        move_robot(objectID["robot"], joint_poses=start_pose)
        if ROBOT_TYPE == "jaco":
            start_coords = robot_coords(objectID["robot"])
            if np.linalg.norm(start_coords[6] - start_pos) > 0.05:
                continue
        # franka
        elif ROBOT_TYPE == "franka":
            # eef_pos = eef_pos(objectID["robot"]) #p.getLinkState(objectID["robot"], 8)[0]
            # print("dist: {} | start pos: {} | eef pos: {}".format(np.linalg.norm(np.array(eef_pos) - np.array(start_pos)), start_pos, eef_pos))
            if np.linalg.norm(np.array(eef_pos(objectID["robot"])) - np.array(start_pos)) > 0.05:
                continue
        goal_pose = random_legalpose(objectID["robot"], pos=goal_pos)
        move_robot(objectID["robot"], joint_poses=goal_pose)

        if ROBOT_TYPE == "jaco":
            goal_coords = robot_coords(objectID["robot"])
            if np.linalg.norm(goal_coords[6] - goal_pos) > 0.05:
                continue
        # franka
        elif ROBOT_TYPE == "franka":
            # eef_pos = eef_pos(objectID["robot"]) #p.getLinkState(objectID["robot"], 8)[0]
            # print("dist: {} | goal pos: {} | eef pos: {}".format(np.linalg.norm(np.array(eef_pos) - np.array(goal_pos)), goal_pos, eef_pos))
            if np.linalg.norm(np.array(eef_pos(objectID["robot"])) - np.array(goal_pos)) > 0.05:
                continue
        return start_pos, goal_pos


def random_SG(robotID, min_dist=0.5):
    """
    Sample a random pair of start and goal robot poses.
    The sampled pair has to be min_dist away in cartesian space.
    """
    while True:
        # Sample random S and G poses.
        start_pose = random_legalpose(robotID)
        goal_pose = random_legalpose(robotID)

        # Check if the sampled S, G pair are far enough away.
        move_robot(robotID, joint_poses=start_pose)
        startEE_xyz = robot_coords(robotID)[-1]
        # startEE_xyz = eef_pos(robotID)
        move_robot(robotID, joint_poses=goal_pose)
        goalEE_xyz = robot_coords(robotID)[-1]
        # goalEE_xyz = eef_pos(robotID)
        distance = np.linalg.norm(startEE_xyz - goalEE_xyz)
        if distance > min_dist:
            return start_pose, goal_pose


def move_laptop(laptopID, laptop_center):
    # posL, _ = p.getBasePositionAndOrientation(laptopID)
    new_pos = [random.uniform(laptop_center[0], laptop_center[0]), random.uniform(laptop_center[1], laptop_center[1]), laptop_center[2]]
    p.resetBasePositionAndOrientation(laptopID, new_pos, p.getBasePositionAndOrientation(laptopID)[1])


def move_robot(robotID, joint_poses=None):
    """
    Move the robot to a legal position above the table.

    Params:
        robotID -- Body unique ID for the robot
        joint_poses -- Desired target poses. If None, choose random legal pose.
    """
    if joint_poses is None:
        joint_poses = random_legalpose(robotID)
    # jaco
    if ROBOT_TYPE == "jaco":
        for joint_index in range(p.getNumJoints(robotID) - 1):
            p.resetJointState(robotID, joint_index + 1, joint_poses[joint_index])
    # franka
    elif ROBOT_TYPE == "franka":
        # for joint_index in list(range(7))+ [9,10]: # minyoung fix - skip panda_link8 and panda_hand - should be fixed to a more general solution - minyoung
        joint_index_mapping = list(range(7)) + [9, 10] # exclude fixed joints: 7, 8, 11
        for joint_index in range(9):
            p.resetJointState(robotID, joint_index_mapping[joint_index], joint_poses[joint_index])
        # p.setJointMotorControlArray(robotID, list(range(7))+[9,10], p.POSITION_CONTROL, joint_poses)
        # p.stepSimulation()


def robot_coords(robotID):
    if ROBOT_TYPE == "jaco":
        states = p.getLinkStates(robotID, range(11))
        coords = np.array([s[0] for s in states])
        return coords[1:8]
    elif ROBOT_TYPE == "franka":
        states = p.getLinkStates(robotID, range(p.getNumJoints(robotID)))
        coords = np.array([s[0] for s in states])
        # preserve :7, 9:11
        # return np.concatenate((coords[:7], coords[9:11]))
        return coords[:9]


def robot_orientations(robotID):
    if ROBOT_TYPE == "jaco":
        states = p.getLinkStates(robotID, range(11))
        orientations = np.array([p.getMatrixFromQuaternion(s[1]) for s in states])
        return orientations[1:8]
    elif ROBOT_TYPE == "franka":
        states = p.getLinkStates(robotID, range(p.getNumJoints(robotID)))
        orientations = np.array([p.getMatrixFromQuaternion(s[1]) for s in states])
        # return np.concatenate((orientations[:7], orientations[9:11]))
        return orientations[:9]

def waypts_to_xyz(robotID, waypts):
    xyz_traj = []
    for waypt in waypts:
        move_robot(robotID, joint_poses=waypt)
        if ROBOT_TYPE == "jaco":
            xyz = robot_coords(robotID)[-1]
        elif ROBOT_TYPE == "franka":
            xyz = eef_pos(robotID)
        xyz_traj.append(xyz)
    return np.array(xyz_traj)


# -- Functions for input space transformations -- #

def raw_features(objectID, waypt):
    """
    Computes raw state space features for a given waypoint.
    ---
    Params:
        waypt -- single waypoint
    Returns:
        raw_features -- list of raw feature values
    """

    # Get relevant objects in the environment.
    posH, _ = p.getBasePositionAndOrientation(objectID["human"])
    posL, _ = p.getBasePositionAndOrientation(objectID["laptop"])
    object_coords = np.array([posH, posL])

    # Get xyz coords and orientations.
    move_robot(objectID["robot"], joint_poses=waypt)
    coords = robot_coords(objectID["robot"])
    orientations = robot_orientations(objectID["robot"])
    # if ROBOT_TYPE == "jaco":
    return np.reshape(np.concatenate((waypt[:7], orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))
    # elif ROBOT_TYPE == "franka":
    #     return np.reshape(np.concatenate((waypt[:9], orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))


def waypts_to_raw(objectID, waypts):
    raw_traj = []
    for waypt in waypts:
        raw = raw_features(objectID, waypt)
        raw_traj.append(raw)
    return np.array(raw_traj)


# -- Functions for visualization -- #

def show_EE_coords(robotID):
    #get current state of EE
    state = p.getLinkState(robotID, 8)
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


# def replay_trajectory(objectID, traj, color=[1, 0, 1]):
#     # raise NotImplementedError # should be fixed for franka
#     xyz_traj = waypts_to_xyz(objectID["robot"], traj)
#     for idx, waypt in enumerate(traj):
#         for jointIndex in range(p.getNumJoints(objectID["robot"]) - 1):
#             p.resetJointState(objectID["robot"], jointIndex + 1, waypt[jointIndex])
#         if idx > 0:
#             p.addUserDebugLine(xyz_traj[idx-1], xyz_traj[idx], lineColorRGB=color, lineWidth=2.0)
#         time.sleep(0.05)

def show_EE_coords(robotID):
    #get current state of EE
    state = p.getLinkState(robotID, 8)
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

def replay_trajectory(objectID, traj, color=[1, 0, 1], buffer_size=0, show_ee_coords=False):
    # raise NotImplementedError # should be fixed for franka
    robotID=objectID["robot"]
    xyz_traj = waypts_to_xyz(robotID, traj)
    for idx, waypt in enumerate(traj):
        if ROBOT_TYPE == "jaco":
            for joint_index in range(p.getNumJoints(robotID) - 1):
                p.resetJointState(robotID, joint_index + 1, waypt[joint_index])
        elif ROBOT_TYPE == "franka":
            joint_index_mapping = list(range(7))
            for joint_index in range(7):
                p.resetJointState(robotID, joint_index_mapping[joint_index], waypt[joint_index])
            p.resetJointState(robotID, 9, 0.04)
            p.resetJointState(robotID, 10, 0.04)
        if idx > 0:
            p.addUserDebugLine(xyz_traj[idx-1], xyz_traj[idx], lineColorRGB=color, lineWidth=10.0)            
            if buffer_size>0:
                for i in range(buffer_size):
                    padding = (i+1)*0.001
                    p.addUserDebugLine(xyz_traj[idx-1]+np.array([0,0,padding]), xyz_traj[idx]+np.array([0,0,padding]), lineColorRGB=color, lineWidth=10.0)
                    p.addUserDebugLine(xyz_traj[idx-1]-np.array([0,0,padding]), xyz_traj[idx]-np.array([0,0,padding]), lineColorRGB=color, lineWidth=10.0)
                    p.addUserDebugLine(xyz_traj[idx-1]+np.array([padding,0,0]), xyz_traj[idx]+np.array([padding,0,0]), lineColorRGB=color, lineWidth=10.0)
                    p.addUserDebugLine(xyz_traj[idx-1]-np.array([padding,0,0]), xyz_traj[idx]-np.array([padding,0,0]), lineColorRGB=color, lineWidth=10.0)
                    p.addUserDebugLine(xyz_traj[idx-1]+np.array([0,padding,0]), xyz_traj[idx]+np.array([0,padding,0]), lineColorRGB=color, lineWidth=10.0)
                    p.addUserDebugLine(xyz_traj[idx-1]-np.array([0,padding,0]), xyz_traj[idx]-np.array([0,padding,0]), lineColorRGB=color, lineWidth=10.0)
        
        if show_ee_coords:
            show_EE_coords(robotID)
        time.sleep(0.05)

def visualize_trajset(objectID, trajs):
    # Visualize all trajectories.
    p.removeAllUserDebugItems()
    xyz_trajs = [waypts_to_xyz(objectID["robot"], traj) for traj in trajs]
    plot_trajset(xyz_trajs)
    time.sleep(0.5)


def plot_trajectory(traj, color=[0, 0, 1]):
    for idx in range(traj.shape[0] - 1):
        p.addUserDebugLine(traj[idx], traj[idx + 1], lineColorRGB=color, lineWidth=2.0)


def plot_trajset(trajs):
    for traj in trajs:
        color = np.random.uniform(0, 1, size=3)
        plot_trajectory(traj, color)

def zview():
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-89.9, cameraPitch=-89.9, cameraTargetPosition=[0.0,0.0,0.0])

def yview():
    p.resetDebugVisualizerCamera(cameraDistance=2.1, cameraYaw=-173.43, cameraPitch=-72.72, cameraTargetPosition=[-0.2,0,0.2])

def xview(view=0):
    # view = np.random.choice(np.arange(4), 1)
    if view == 0:
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-270, cameraPitch=-15, cameraTargetPosition=[0,0,0.5])
    elif view == 1:
        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=-340, cameraPitch=-15, cameraTargetPosition=[0,0,0.5])
    elif view == 2:
        p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=-225, cameraPitch=-7, cameraTargetPosition=[0,0,0.5])
    elif view == 3:
        p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=-385, cameraPitch=-15, cameraTargetPosition=[0,0,0.5])