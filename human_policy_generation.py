### --- Human Agent --- ###
import numpy as np
import json
from typing import Any, List, Tuple, Dict, Union
import os
from scipy.interpolate import splev, splrep
from warp_trajectory import warp_matrix

# class HumanLikePolicy:
#     def __init__(self, task_name:str, demo_file:str, episode_len:int) -> None:

#         # with open(demo_file) as f:
#         #     self.json_file = json.load(f)

#         # if task_name != self.json_file["task_name"]:
#         #     raise ValueError(f"Task name: {task_name} does not match task name in demo file: {self.json_file['task_name']}")

#         # Hardcoded for now, will fix later
#         if task_name == 'pick_and_place':
#             self.agent = PickAndPlaceHumanAgent(HARDCODE_PATH, 5, episode_len, start=100, z_start = 260, end=-1)
        
#         else:
#             raise NotImplementedError(f"Task: {task_name} is not implemented yet.")
#         else:
#             raise NotImplementedError(f"Task: {task_name} is not implemented yet.")
        
#     def reset(self, desired_goal: np.ndarray, achieved_goal: np.ndarray) -> None:
#         self.agent.reset(object_location = achieved_goal, goal_location = desired_goal)
#     def reset(self, desired_goal: np.ndarray, achieved_goal: np.ndarray) -> None:
#         self.agent.reset(object_location = achieved_goal, goal_location = desired_goal)

#     def __call__(self, step: int) -> np.ndarray:
#         return self.agent.get_goal_pose(step)
#     def __call__(self, step: int) -> np.ndarray:
#         return self.agent.get_goal_pose(step)
    
# # Human Agent for the pick and place task
# class PickAndPlaceHumanAgent:
#     def __init__(self, filename, start_steps, total_steps, start=0, z_start=0, end=-1, 
#                  offset = [0, 0, 0, 0]):
#         """
#         The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
#         :param env:         training enviroment
#         :param file_path:   file path of the txt file the oculus recording data is stored in
#         :param start_steps: The number of steps the human agent will spend at the begining of the 
#                             simulation trying to reach its initital postiion (first recorded point)
#         :param total_steps: total number of timesteps per episode
#         :param start:       The index of the oculus recorded data to start at
#         :param z_start:     The index at which scaling of the z-axis (vertical motion) begins.
#         :param end:         The index of the oculus recording to stop at
#         :param size:        Size of the memory buffer. How many total runs to be stored
#         :param offset:      Offset to add to way points from the oculus 
#         """
# # Human Agent for the pick and place task
# class PickAndPlaceHumanAgent:
#     def __init__(self, filename, start_steps, total_steps, start=0, z_start=0, end=-1, 
#                  offset = [0, 0, 0, 0]):
#         """
#         The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
#         :param env:         training enviroment
#         :param file_path:   file path of the txt file the oculus recording data is stored in
#         :param start_steps: The number of steps the human agent will spend at the begining of the 
#                             simulation trying to reach its initital postiion (first recorded point)
#         :param total_steps: total number of timesteps per episode
#         :param start:       The index of the oculus recorded data to start at
#         :param z_start:     The index at which scaling of the z-axis (vertical motion) begins.
#         :param end:         The index of the oculus recording to stop at
#         :param size:        Size of the memory buffer. How many total runs to be stored
#         :param offset:      Offset to add to way points from the oculus 
#         """
        
#         self.z_start = z_start - start # Adjust z_start wrt start
#         self.start_steps = start_steps
#         self.total_steps = total_steps
#         self.offset = np.array(offset)
#         data = []
#         self.z_start = z_start - start # Adjust z_start wrt start
#         self.start_steps = start_steps
#         self.total_steps = total_steps
#         self.offset = np.array(offset)
#         data = []
        
#         # Read in the data from the oculus and organize into a list of lists.
#         # Each element of the list corresponds to a single oculus time step.
#         with open(filename) as f:
#             lines = f.readlines()
#         # Read in the data from the oculus and organize into a list of lists.
#         # Each element of the list corresponds to a single oculus time step.
#         with open(filename) as f:
#             lines = f.readlines()

#         for line in lines:
#             index, position, finger = line[:-1].split('\t')
#             position = np.array(position[1:-1].split(', ')).astype(np.float64)
#             dataline = [float(index), position, float(finger)]
#             data.append(dataline)
#         for line in lines:
#             index, position, finger = line[:-1].split('\t')
#             position = np.array(position[1:-1].split(', ')).astype(np.float64)
#             dataline = [float(index), position, float(finger)]
#             data.append(dataline)
        
#         # Clip the data to the start and end indices
#         self.data = data[start:end]
#         # Clip the data to the start and end indices
#         self.data = data[start:end]
        
#     def reset(self, object_location, goal_location):
#         """
#         Resets the ajsutments for the start and goal locations.
#         Needs to be run every time the enviroment is reset.
#         :param object_location: The start location of the object (the block)
#         :param goal_location:   The location of the goal                       
#         """
#         # The locations of the object and goal in the oculus sim.
#         rec_obj_location = np.array([0.1, 0.1, 0.02])
#         rec_goal_location = np.array([-0.1, -0.1, 0.12])
#     def reset(self, object_location, goal_location):
#         """
#         Resets the ajsutments for the start and goal locations.
#         Needs to be run every time the enviroment is reset.
#         :param object_location: The start location of the object (the block)
#         :param goal_location:   The location of the goal                       
#         """
#         # The locations of the object and goal in the oculus sim.
#         rec_obj_location = np.array([0.1, 0.1, 0.02])
#         rec_goal_location = np.array([-0.1, -0.1, 0.12])
        
#         # Find the adjustments for the simulation instance
#         self.adjust_mult = (object_location - goal_location)/(rec_obj_location - rec_goal_location)
#         self.adjust_bias = object_location - self.adjust_mult*rec_obj_location
#         # Find the adjustments for the simulation instance
#         self.adjust_mult = (object_location - goal_location)/(rec_obj_location - rec_goal_location)
#         self.adjust_bias = object_location - self.adjust_mult*rec_obj_location
          
#     def get_goal_pose(self, step: int) -> np.ndarray:
#         """
#         Returns an action from the human agent given a simulation step 
#         :param step:  Pybullet simulation step
#         :return:      Panda EE Action (4x1 np array)
#         """
#         # Find i, the index of the oculus recording that corresponds 
#         # to the given simulation step
#     def get_goal_pose(self, step: int) -> np.ndarray:
#         """
#         Returns an action from the human agent given a simulation step 
#         :param step:  Pybullet simulation step
#         :return:      Panda EE Action (4x1 np array)
#         """
#         # Find i, the index of the oculus recording that corresponds 
#         # to the given simulation step
        
#         # Durring the first start_steps, have the arm just travel to the 
#         # starting location. Then, evenly distribute the remaining sim
#         # steps amoung the oculus recording steps.
#         if step < self.start_steps:
#             i = 0
#         else:
#             i = round((step-self.start_steps)*len(self.data)/(self.total_steps - self.start_steps))
#         # Durring the first start_steps, have the arm just travel to the 
#         # starting location. Then, evenly distribute the remaining sim
#         # steps amoung the oculus recording steps.
#         if step < self.start_steps:
#             i = 0
#         else:
#             i = round((step-self.start_steps)*len(self.data)/(self.total_steps - self.start_steps))
               
#         # Adjust the recorded goal location to align the coordinate frames.
#         goal_location = self.data[i][1]
#         goal_location = np.array([goal_location[2], -goal_location[0], goal_location[1]])
#         # Adjust the recorded goal location to align the coordinate frames.
#         goal_location = self.data[i][1]
#         goal_location = np.array([goal_location[2], -goal_location[0], goal_location[1]])
        
#         # Adjust the goal locations to scale and shift the EE motions to match
#         # the sim's oject start and goal locations. 
#         # If the time step is before z_start, only adjust the x and y corrdinates. 
#         if i > self.z_start:
#             goal_location = goal_location*self.adjust_mult + self.adjust_bias
#         else:
#             goal_location[0:2] = goal_location[0:2]*self.adjust_mult[0:2] + self.adjust_bias[0:2]
#         # Adjust the goal locations to scale and shift the EE motions to match
#         # the sim's oject start and goal locations. 
#         # If the time step is before z_start, only adjust the x and y corrdinates. 
#         if i > self.z_start:
#             goal_location = goal_location*self.adjust_mult + self.adjust_bias
#         else:
#             goal_location[0:2] = goal_location[0:2]*self.adjust_mult[0:2] + self.adjust_bias[0:2]
                
#         # Combine the goal location and goal gripper location to get the goal_postion
#         goal_position = np.zeros(4)
#         goal_position[0:3] = goal_location
#         goal_position[3] = 2*self.data[i][2]
#         # Combine the goal location and goal gripper location to get the goal_postion
#         goal_position = np.zeros(4)
#         goal_position[0:3] = goal_location
#         goal_position[3] = 2*self.data[i][2]
        
#         return goal_position
#         return goal_position


# Fancy Version. I'm not gonna bother with it just yet. 
class HumanLikePolicy:
    def __init__(self, task_name:str, demo_file:str, total_steps:int) -> None:
        self.TASK_NAME = task_name
        self.DEMO_FILE = demo_file

        with open(self.DEMO_FILE) as f:
            json_data = json.load(f)

        if task_name != json_data["task_name"]:
            raise ValueError(f"Task name: {task_name} does not match task name in demo file: {json_data['task_name']}")

        if task_name in ['pick_and_place', 'push', 'stack']:
            self.agent = PickAndPlaceHumanAgent(json_data, total_steps)
        
        else:
            raise NotImplementedError(f"Task: {task_name} is not implemented yet.")
        
    def reset(self, env) -> None:
        self.agent.reset(env)

    def __call__(self, step: int) -> np.ndarray:
        return self.agent.get_goal_pose(step)
    
# Human Agent for the pick and place task
class PickAndPlaceHumanAgent:

    def __init__(self, json_data: Dict[str, any], total_steps: int):
        """
        The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
        :param json_data:   The json file containing the header info and data for the oculus recording:
            data:        The oculus recording data
            start_steps: The number of steps the human agent will spend at the begining of the 
                            simulation trying to reach its initital postiion (first recorded point)
            total_steps: total number of timesteps per episode
            start:       The index of the oculus recorded data to start at
            z_start:     The index at which scaling of the z-axis (vertical motion) begins.
            end:         The index of the oculus recording to stop at
            size:        Size of the memory buffer. How many total runs to be stored
            offset:      Offset to add to way points from the oculus 
        """
        
        self.json_data = json_data
        self.total_steps = total_steps

        # Exctract data: Each element of the list corresponds to a single oculus time step.
        self.data: List[Tuple(float, List[float], float)] = self.json_data["data"]

        ts, positions, gs = zip(*self.data)
        xs, ys, zs = zip(*positions)

        # Represent the data as splines
        self.splines = [splrep(ts, xs), splrep(ts, ys), splrep(ts, zs), splrep(ts, gs)]

        self.anchor_points: List[Tuple[str, int, List[float]]]= self.json_data["anchor_points"]
        self.anchor_points[-1][1] = len(self.data) # Set the ending time_step of the last anchor point to the end of the recording
        # self.anchor_points: List[Tuple[str, int, List[float]]] = [("object", 240, [0.1, 0.1, 0.02]), ("target", len(self.data), [-0.1, -0.1, 0.12])]

        self.end_wait:int = self.json_data["end_wait_portion"]*total_steps
        self.offset:np.ndarray = np.array(self.json_data["offset"])
        
        
    def reset(self, env):
        """
        Resets the ajsutments for the start and goal locations.
        Needs to be run every time the enviroment is reset.
        :param object_location: The start location of the object (the block)
        :param goal_location:   The location of the goal                       
        """
        
        # Find the warps for the simulation instance
        self.warps: List[np.ndarray(size=(4,4))] = []

        # Defualt first anchor point is the endeffector location at the start of the episode
        anchor_locations_rec: List[np.ndarray] = [np.array(self.data[0][1])]
        anchor_locations_env: List[np.ndarray] = [env.env.robot.get_ee_position()]
        self.anchor_indecies: List[int] = [0]

        for anchor_point in self.anchor_points:
            anchor_name, anchor_index, anchor_location_rec = anchor_point
            anchor_locations_rec.append(np.array(anchor_location_rec))
            anchor_locations_env.append(env.env.sim.get_base_position(anchor_name)[0:3])
            self.anchor_indecies.append(anchor_index)

        for i in range(len(anchor_locations_rec)-1):
            # print('i:', i)
            # print('rec:', [anchor_locations_rec[i], anchor_locations_rec[i+1]])
            # print('goal:', [anchor_locations_env[i], anchor_locations_env[i+1]])
            self.warps.append(warp_matrix(rec_start=anchor_locations_rec[i], 
                                          rec_end=anchor_locations_rec[i+1], 
                                          new_start=anchor_locations_env[i], 
                                          new_end=anchor_locations_env[i+1]))
            

    def get_goal_pose(self, step: int) -> np.ndarray:
        """
        Returns an action from the human agent given a simulation step 
        :param step:  Pybullet simulation step
        :return:      Panda EE Action (4x1 np array)
        """
        # Find i, the index of the oculus recording that corresponds 
        # to the given simulation step
        
        # Durring the first start_steps, have the arm just travel to the 
        # starting location. Then, evenly distribute the remaining sim
        # steps amoung the oculus recording steps.
        if step > self.total_steps - self.end_wait:
            t = len(self.data) - 1

        else:
            t = step*(len(self.data)-1)/(self.total_steps-self.end_wait)
               
        # Find the goal location from the splines
        goal_location = np.array([splev(t, self.splines[i]) for i in range(4)])
        
        # Adjust the goal locations to scale and shift the EE motions to match
        # the sim's oject start and goal locations. 
        # If the time step is before z_start, only adjust the x and y corrdinates. 
        adjust_index = 0
        for i in range(len(self.warps)):
            if t > self.anchor_indecies[i]:
                adjust_index = i

        # print('t:', t, 'cutoffs', self.anchor_indecies, "adjust_index", adjust_index)

        goal_location_homogeneous = np.ones(4)
        goal_location_homogeneous[0:3] = goal_location[0:3]
        warped_goal_location = self.warps[adjust_index] @ goal_location_homogeneous
        goal_location[0:3] = warped_goal_location[0:3]
        # goal_location[0:3] = goal_location[0:3]*self.adjust_mults[adjust_index] + self.adjust_biases[adjust_index]
        # if t > self.z_start:
        #     goal_location[0:3] = goal_location[0:3]*self.adjust_mult + self.adjust_bias
        # else:
        #     goal_location[0:2] = goal_location[0:2]*self.adjust_mult[0:2] + self.adjust_bias[0:2]
        
        return goal_location
    
def generate_json_file(save_file: str, oculus_file: str, task_name:str, anchor_points: List[Tuple[str, int, List[float]]],
                       start:int=0, end:int=None, end_wait_portion:float=0, offset:List[float] = [0, 0, 0, 0]):
    """ Generates a JSON file containing the header info and data for the oculus recording:
    :param save_file:    file path of the json file to save the data to
    :param oculus_file:   file path of the txt file the oculus recording data is stored in
    :param start_steps: The number of steps the human agent will spend at the begining of the 
                        simulation trying to reach its initital postiion (first recorded point)
    :param start:       The index of the oculus recorded data to start at
    :param z_start:     The index at which scaling of the z-axis (vertical motion) begins.
    :param end:         The index of the oculus recording to stop at
    :param size:        Size of the memory buffer. How many total runs to be stored
    :param offset:      Offset to add to way points from the oculus 
    """
    

    # self.rec_obj_location = np.array([0.1, 0.1, 0.02])
    # self.rec_goal_location = np.array([-0.1, -0.1, 0.12])

    data = []
    
    # Read in the data from the oculus and organize into a list of lists.
    # Each element of the list corresponds to a single oculus time step.
    with open(oculus_file) as f:
        lines = f.readlines()

    for line in lines:
        index, position, finger = line[:-1].split('\t')
        position = np.array(position[1:-1].split(', ')).astype(np.float64)
        position = [position[2], -position[0], position[1]] # Adjust for sim coordinate frame
        dataline = [float(index)-start, position, 2*float(finger)] # Adjust for sim gripper
        data.append(dataline)
    
    # Clip the data to the start and end indices
    if end is None:
        data = data[start:]
    else:
        data = data[start:end]

    json_data = {"task_name": task_name,
                 "end_wait_portion": end_wait_portion,
                 "offset": offset,
                 "anchor_points": anchor_points,
                 "data": data}
    
    with open(save_file, 'w') as f:
        json.dump(json_data, f)


class replay_oculus_recording:
    def __init__ (self, oculus_file:str, num_steps:int, start:int = 0, end:int = None) -> None:   
        data = []
        self.total_steps = num_steps
    
        # Read in the data from the oculus and organize into a list of lists.
        # Each element of the list corresponds to a single oculus time step.
        with open(oculus_file) as f:
            lines = f.readlines()

        for line in lines:
            index, position, finger = line[:-1].split('\t')
            position = np.array(position[1:-1].split(', ')).astype(np.float64)
            position = [position[2], -position[0], position[1]] # Adjust for sim coordinate frame
            dataline = [float(index)-start, position, 2*float(finger)] # Adjust for sim gripper
            data.append(dataline)
        
        if end is None:
            self.data = data[start:]
        else:
            self.data = data[start:end]

        ts, positions, gs = zip(*self.data)
        xs, ys, zs = zip(*positions)

        # Represent the data as splines
        self.splines = [splrep(ts, xs), splrep(ts, ys), splrep(ts, zs), splrep(ts, gs)]
        

    def __call__(self, step:int) -> None:
        t = step*(len(self.data)-1)/self.total_steps
        print('t: ', t)
               
        # Find the goal location from the splines
        return np.array([splev(t, self.splines[i]) for i in range(4)])

        

if __name__ == "__main__":
    from my_sim_env import CustomEnv # for evaluating the policy
    import time
    if os.name == 'nt':
        HARDCODE_PATH = "C:/Research/Transformers/SingleDemoACT/test59_edited.txt"
    else:
        # HARDCODE_PATH = "/home/robert/Research/Transformers/test59.txt"
        HARDCODE_PATH = "/home/aigeorge/research/SingleDemoACT/test59_edited.txt"

    CAMERA_NAME: str = 'top'
    TASK_NAME: str = 'stack'
    env = CustomEnv(task_name=TASK_NAME, inject_noise=False, camera_names=[CAMERA_NAME], onscreen_render=True, transparent_arm=False) # create the environment

    for i in range(0):
        oculus_replay = replay_oculus_recording(oculus_file='/home/aigeorge/research/SingleDemoACT/Oculus_Data/test66_edited.txt', num_steps=100)
        env.env.sim.set_base_pose("target1", np.array([-0.1, 0, 0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
        env.env.sim.set_base_pose("target2", np.array([-0.1, 0, 0.06]), np.array([0.0, 0.0, 0.0, 1.0]))
        env.env.sim.set_base_pose("object1", np.array([0.1, -0.1, 0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
        env.env.sim.set_base_pose("object2", np.array([0.1, 0.1, 0.02]), np.array([0.0, 0.0, 0.0, 1.0]))

        for i in range(100):
            goal_position = oculus_replay(i)
            input()
            env.step_pos(goal_position)

    if os.name == 'nt':
        HARDCODE_PATH = "C:/Research/Transformers/SingleDemoACT/Oculus_Data/test66.txt"
    else:
        HARDCODE_PATH = "/home/aigeorge/research/SingleDemoACT/Oculus_Data/test66_edited.txt"

    ANCHOR_POINTS: List[Tuple[str, int, List[float]]] = [("object1", 170, [0.1, -0.1, 0.02]), ("target1", 350, [-0.1, 0, 0.02]), ("object2", 580, [0.1, 0.1, 0.02]), ("target2", 830, [-0.1, 0, 0.06])] # oculus 69
    ANCHOR_POINTS: List[Tuple[str, int, List[float]]] = [("object1", 180, [0.1, -0.1, 0.02]), ("target1", 465, [-0.1, 0, 0.02]), ("object2", 710, [0.1, 0.1, 0.02]), ("target2", 950, [-0.1, 0, 0.06])] # oculus 66
    # Generate a json file for the pick and place task:
    generate_json_file(save_file='stack_delete.json', 
                       oculus_file=HARDCODE_PATH, 
                       task_name=TASK_NAME, 
                       anchor_points=ANCHOR_POINTS,
                       end_wait_portion=0, #0.1,
                       start=0, #100 
                       end=-1)
    
    generate_json_file(save_file='stack.json', 
                       oculus_file=HARDCODE_PATH, 
                       task_name=TASK_NAME, 
                       anchor_points=ANCHOR_POINTS,
                       end_wait_portion=0, #0.1,
                       start=0, #100, 
                       end=-1)
    
    policy = HumanLikePolicy(TASK_NAME, 'stack_delete.json', env.MAX_TIME_STEPS) # create the policy

    # Base recording play:
    for _ in range(0):
        obs: Dict[str, np.ndarray] = env.reset() # reset the environment
        env.env.sim.set_base_pose("target1", np.array([-0.1, 0, 0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
        env.env.sim.set_base_pose("target2", np.array([-0.1, 0, 0.06]), np.array([0.0, 0.0, 0.0, 1.0]))
        env.env.sim.set_base_pose("object1", np.array([0.1, -0.1, 0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
        env.env.sim.set_base_pose("object2", np.array([0.1, 0.1, 0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
        env.env.task.goal = np.array([-0.1, 0, 0.02,-0.1, 0, 0.06])
        time.sleep(2)

        policy.reset(env) # reset the policy
        for step in range(env.MAX_TIME_STEPS):
            goal_position: np.ndarray = policy(step)
            # time.sleep(0.01)
            print('goal_position', goal_position)
            env.step_pos(goal_position)

    
    policy = HumanLikePolicy(TASK_NAME, 'stack.json', env.MAX_TIME_STEPS) # create the policy


    # Test the policy:
    for _ in range(10):
        obs: Dict[str, np.ndarray] = env.reset() # reset the environment
        policy.reset(env) # reset the policy
        for step in range(env.MAX_TIME_STEPS):
            goal_position: np.ndarray = policy(step)
            print('goal_position:', goal_position)
            input()
            env.step_pos(goal_position)

    # if os.name == 'nt':
    #     HARDCODE_PATH = "C:/Research/Transformers/SingleDemoACT/test59_edited.txt"
    # else:
    #     # HARDCODE_PATH = "/home/robert/Research/Transformers/test59.txt"
    #     HARDCODE_PATH = "/home/aigeorge/research/SingleDemoACT/test59_edited.txt"

    # CAMERA_NAME: str = 'top'
    # TASK_NAME: str = 'pick_and_place'
    # ANCHOR_POINTS: List[Tuple[str, int, List[float]]] = [("object", 240, [0.1, 0.1, 0.02]), ("target", 1000, [-0.1, -0.1, 0.12])]
    # REC_GOAL: np.ndarray = np.array([-0.1, -0.1, 0.12])
    # REC_OBJ: np.ndarray = np.array([0.1, 0.1, 0.02])
    # env = CustomEnv(TASK_NAME, False, [CAMERA_NAME], True) # create the environment
    # # Generate a json file for the pick and place task:
    # generate_json_file(save_file='pick_and_place_base_edit.json', 
    #                    oculus_file=HARDCODE_PATH, 
    #                    task_name=TASK_NAME, 
    #                    anchor_points=ANCHOR_POINTS,
    #                    end_wait_portion=0, #0.1,
    #                    start=0, #100 
    #                    end=-1)
    
    # generate_json_file(save_file='pick_and_place_adjust_edit.json', 
    #                    oculus_file=HARDCODE_PATH, 
    #                    task_name=TASK_NAME, 
    #                    anchor_points=ANCHOR_POINTS,
    #                    end_wait_portion=0.1,
    #                    start=100, 
    #                    end=-1)
    
    # policy = HumanLikePolicyJSON(TASK_NAME, 'pick_and_place_base_edit.json', env.MAX_TIME_STEPS) # create the policy

    # # Base recording play:
    # for _ in range(5):
    #     obs: Dict[str, np.ndarray] = env.reset() # reset the environment
    #     env.env.sim.set_base_pose("target", REC_GOAL, np.array([0.0, 0.0, 0.0, 1.0]))
    #     env.env.sim.set_base_pose("object", REC_OBJ, np.array([0.0, 0.0, 0.0, 1.0]))
    #     env.env.task.goal = REC_GOAL

    #     policy.reset(env) # reset the policy
    #     for step in range(env.MAX_TIME_STEPS):
    #         goal_position: np.ndarray = policy(step)
    #         print(goal_position)
    #         env.step_pos(goal_position)

    
    # policy = HumanLikePolicyJSON(TASK_NAME, 'pick_and_place_adjust_edit.json', env.MAX_TIME_STEPS) # create the policy


    # # Test the policy:
    # for _ in range(10):
    #     obs: Dict[str, np.ndarray] = env.reset() # reset the environment
    #     policy.reset(env) # reset the policy
    #     for step in range(env.MAX_TIME_STEPS):
    #         goal_position: np.ndarray = policy(step)
    #         print(goal_position)
    #         env.step_pos(goal_position)