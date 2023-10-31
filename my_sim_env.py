import numpy as np
from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv, PandaStackEnv
from typing import Tuple, List, Dict, Any
import time
import matplotlib.pyplot as plt
import cv2
from test import plot_action

AVAILABLE_SIM_NAMES: List[str] = ['pick_and_place', 'push', 'stack'] 
AVAILABLE_CAM_NAMES: List[str] = ['top', 'front', 'side-left', 'side-right', 'ee', 'isotropic']

CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480

MAX_POSITION: np.ndarray = np.array([0.35, 0.35, 0.7, 0.08])
MIN_POSITION: np.ndarray = np.array([-0.35, -0.35, 0.0, 0])

### output_obs: dict['success', 'position (4)', 'velocity(3)', 'desired_goal', 'achieved_goal']

class OUNoise:
    def __init__(self, theta, sigma, size, burn_in = 50):
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.x = np.zeros(size)

        for _ in range(burn_in):
            self.step()

    def step(self):
        dx = self.theta*(0-self.x) + self.sigma*np.random.randn(self.size)
        self.x += dx
        return self.x

class CustomEnv:
    def __init__(self, task_name:str, inject_noise:bool, camera_names: List[str], onscreen_render:bool, random_start:bool = False, K_POS:float = 15.0, K_GRIP:float = 3.0, transparent_arm:bool = False) -> None:
        self.task_name: str = task_name
        self.inject_noise: bool = inject_noise
        self.camera_names: List[str] = camera_names
        self.render: bool = onscreen_render
        self.random_start: bool = random_start

        self.K_POS = K_POS
        self.K_GRIP = K_GRIP

        self.DT = 0.04 # 25 Hz simulation

        self.render_goal = False

        # check to makes sure the specified task is a valid sim task:
        assert task_name in AVAILABLE_SIM_NAMES, f"TASK_NAME: {task_name} is not a valid sim task. Please choose from: {AVAILABLE_SIM_NAMES}"

        # check to makes sure the specified camera names are valid:
        for cam_name in camera_names:
            assert cam_name in AVAILABLE_CAM_NAMES, f"cam_name: {cam_name} is not a valid camera name. Please choose from: {AVAILABLE_CAM_NAMES}"

        if task_name in ['pick_and_place', 'push']:
            self.env = PandaPickAndPlaceEnv(render_mode = "rgb_array",
                                            reward_type = "sparse",
                                            control_type = "ee",
                                            renderer = "OpenGL",
                                            render_width = 640,
                                            render_height = 480,
                                            render_target_position = None,
                                            render_distance = 1.4,
                                            render_yaw = 45,
                                            render_pitch = -30,
                                            render_roll = 0,) # task enviroment # task enviroment
            self.MAX_TIME_STEPS: int = 50
            self.MAX_REWARD: int = 0
            self.reward = -1

        elif task_name == 'stack':
            self.env = PandaStackEnv(render_mode = "rgb_array",
                                            reward_type = "sparse",
                                            control_type = "ee",
                                            renderer = "OpenGL",
                                            render_width = 640,
                                            render_height = 480,
                                            render_target_position = None,
                                            render_distance = 1.4,
                                            render_yaw = 45,
                                            render_pitch = -30,
                                            render_roll = 0,) # task enviroment # task enviroment
            self.MAX_TIME_STEPS: int = 100
            self.MAX_REWARD: int = 0
            self.reward = -1

        else:
            raise NotImplementedError(f"Task: {task_name} is not implemented yet.")
        
        
        # Rendering Schenanagins:
        # Set the arm to be transparent:
        if transparent_arm:
            for i in range(9):
                self.env.sim.physics_client.changeVisualShape(self.env.sim._bodies_idx['panda'], i, rgbaColor=[0, 0, 0, 0])
            self.env.sim.physics_client.changeVisualShape(self.env.sim._bodies_idx['panda'], -1, rgbaColor=[0, 0, 0, 0])

            self.env.sim.physics_client.changeVisualShape(self.env.sim._bodies_idx['table'], -1, rgbaColor=[0.5, 0.5, 0.5, 1])

        if self.task_name in ["pick_and_place", "push"]:
            self.env.sim.physics_client.changeVisualShape(self.env.sim._bodies_idx['target'], -1, rgbaColor=[0, 0, 1, 0.3])
        
        # # Setup the camera renders:
        # self.camera_renders = {}
        # if "top" in camera_names:
        #     self.camera_renders["top"] = lambda :self.env.sim.render(width = CAMERA_WIDTH*2, 
        #                                                         height= CAMERA_HEIGHT*2,
        #                                                         target_position = None,
        #                                                         distance = 1,
        #                                                         yaw = 0,
        #                                                         pitch= -90,
        #                                                         roll = 0)[CAMERA_HEIGHT//2:CAMERA_HEIGHT*3//2, CAMERA_WIDTH//2:CAMERA_WIDTH*3//2, :]
        #     if self.render:
        #         cv2.namedWindow("top", cv2.WINDOW_NORMAL)
        #         cv2.resizeWindow("top", CAMERA_WIDTH, CAMERA_HEIGHT)

        # if "front" in camera_names:
        #     self.camera_renders["front"] = lambda :self.env.sim.render(width = CAMERA_WIDTH*2, 
        #                                                         height= CAMERA_HEIGHT*2,
        #                                                         target_position = None,
        #                                                         distance = 1,
        #                                                         yaw = 90,
        #                                                         pitch= 0,
        #                                                         roll = 0)[CAMERA_HEIGHT//6:CAMERA_HEIGHT*7//6, CAMERA_WIDTH//2:CAMERA_WIDTH*3//2, :]
        #     if self.render:
        #         cv2.namedWindow("front", cv2.WINDOW_NORMAL)
        #         cv2.resizeWindow("front", CAMERA_WIDTH, CAMERA_HEIGHT)

        #     print('using front camera')

        # if "side-left" in camera_names:
        #     self.camera_renders["side-left"] = lambda :self.env.sim.render(width = CAMERA_WIDTH*2, 
        #                                                         height= CAMERA_HEIGHT*2,
        #                                                         target_position = None,
        #                                                         distance = 1,
        #                                                         yaw = 0,
        #                                                         pitch= 0,
        #                                                         roll = 0)[CAMERA_HEIGHT//6:CAMERA_HEIGHT*7//6, CAMERA_WIDTH//2:CAMERA_WIDTH*3//2, :]
        #     if self.render:
        #         cv2.namedWindow("side-left", cv2.WINDOW_NORMAL)
        #         cv2.resizeWindow("side-left", CAMERA_WIDTH, CAMERA_HEIGHT)

        #     print('using left camera')
        
        # if "side-right" in camera_names:
        #     self.camera_renders["side-right"] = lambda :self.env.sim.render(width = CAMERA_WIDTH*2, 
        #                                                         height= CAMERA_HEIGHT*2,
        #                                                         target_position = None,
        #                                                         distance = 1,
        #                                                         yaw = 180,
        #                                                         pitch= 0,
        #                                                         roll = 0)[CAMERA_HEIGHT//6:CAMERA_HEIGHT*7//6, CAMERA_WIDTH//2:CAMERA_WIDTH*3//2, :]
        #     if self.render:
        #         cv2.namedWindow("side-right", cv2.WINDOW_NORMAL)
        #         cv2.resizeWindow("side-right", CAMERA_WIDTH, CAMERA_HEIGHT)

        # if "ee" in camera_names:
        #     self.camera_renders["ee"] = lambda :self.env.sim.render(width = CAMERA_WIDTH,
        #                                                         height= CAMERA_HEIGHT,
        #                                                         target_position = np.array([self.env_dict["observation"][0], self.env_dict["observation"][1], 0]),
        #                                                         distance = self.env_dict["observation"][2] + 0.25,
        #                                                         yaw = -90,
        #                                                         pitch= -45,
        #                                                         roll = 0)
        #     if self.render:
        #         cv2.namedWindow("ee", cv2.WINDOW_NORMAL)
        #         cv2.resizeWindow("ee", CAMERA_WIDTH, CAMERA_HEIGHT)
        
        # Setup the camera renders:
        self.camera_renders = {}
        if "top" in camera_names:
            self.camera_renders["top"] = lambda :self.env.sim.render(width = CAMERA_WIDTH, 
                                                                height= CAMERA_HEIGHT,
                                                                target_position = None,
                                                                distance = 0.5,
                                                                yaw = 0,
                                                                pitch= -90,
                                                                roll = 0)
            if self.render:
                cv2.namedWindow("top", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("top", CAMERA_WIDTH, CAMERA_HEIGHT)

        if "front" in camera_names:
            self.camera_renders["front"] = lambda :self.env.sim.render(width = CAMERA_WIDTH, 
                                                                height= CAMERA_HEIGHT,
                                                                target_position = None,
                                                                distance = 0.5,
                                                                yaw = 90,
                                                                pitch= 0,
                                                                roll = 0)
            if self.render:
                cv2.namedWindow("front", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("front", CAMERA_WIDTH, CAMERA_HEIGHT)

            print('using front camera')

        if "side-left" in camera_names:
            self.camera_renders["side-left"] = lambda :self.env.sim.render(width = CAMERA_WIDTH, 
                                                                height= CAMERA_HEIGHT,
                                                                target_position = None,
                                                                distance = 0.5,
                                                                yaw = 0,
                                                                pitch= 0,
                                                                roll = 0)
            if self.render:
                cv2.namedWindow("side-left", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("side-left", CAMERA_WIDTH, CAMERA_HEIGHT)

            print('using left camera')
        
        if "side-right" in camera_names:
            self.camera_renders["side-right"] = lambda :self.env.sim.render(width = CAMERA_WIDTH, 
                                                                height= CAMERA_HEIGHT,
                                                                target_position = None,
                                                                distance = 0.5,
                                                                yaw = 180,
                                                                pitch= 0,
                                                                roll = 0)
            if self.render:
                cv2.namedWindow("side-right", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("side-right", CAMERA_WIDTH, CAMERA_HEIGHT)

        if "ee" in camera_names:
            self.camera_renders["ee"] = lambda :self.env.sim.render(width = CAMERA_WIDTH,
                                                                height= CAMERA_HEIGHT,
                                                                target_position = np.array([self.env_dict["observation"][0], self.env_dict["observation"][1], 0]),
                                                                distance = self.env_dict["observation"][2] + 0.25,
                                                                yaw = -90,
                                                                pitch= -45,
                                                                roll = 0)
            if self.render:
                cv2.namedWindow("ee", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("ee", CAMERA_WIDTH, CAMERA_HEIGHT)
        
        if "isotropic" in camera_names:
            self.camera_renders["isotropic"] = lambda :self.env.sim.render(width = CAMERA_WIDTH, 
                                                                height= CAMERA_HEIGHT,
                                                                target_position = None,
                                                                distance = 0.75,
                                                                yaw = 45,
                                                                pitch= -45,
                                                                roll = 0)

        self.env_dict: Dict[str, np.ndarray] = self.env.reset()[0]

    def get_obs(self) -> Dict[str, np.ndarray]:
        if self.task_name in ["pick_and_place", "push"]:
            success: bool = np.linalg.norm(self.env_dict["achieved_goal"] - self.env_dict["desired_goal"]) < 0.02
        elif self.task_name == "stack":
            success: bool = np.linalg.norm(self.env_dict["achieved_goal"][:3] - self.env_dict["desired_goal"][:3]) < 0.02 \
                and np.linalg.norm(self.env_dict["achieved_goal"][3:] - self.env_dict["desired_goal"][3:]) < 0.02
            
        position: np.ndarray = np.concatenate((self.env_dict["observation"][:3], [self.env_dict["observation"][6]]))
        velocity: np.ndarray = self.env_dict["observation"][3:6]
        rendered_images = {}
        if self.render_goal:
            display_images = {}
        for camera in self.camera_names:
            rendered_images[camera] = self.camera_renders[camera]()
            if self.render_goal:
                self.env.sim.create_box(body_name="goal_finger_1", 
                                        half_extents = np.array([0.005, 0.0025, 0.005]),
                                        mass = 0,
                                        position = self.env_dict["observation"][:3] + np.array([0.0, self.env_dict["observation"][6], 0.0]),
                                        rgba_color=[1, 0, 0, 1],
                                        ghost=True)
                
                self.env.sim.create_box(body_name="goal_finger_2", 
                                        half_extents = np.array([0.005, 0.0025, 0.005]),
                                        mass = 0,
                                        position = self.env_dict["observation"][:3] - np.array([0.0, self.env_dict["observation"][6], 0.0]),
                                        rgba_color=[1, 0, 0, 1],
                                        ghost=True)
                display_images[camera] = self.camera_renders[camera]()
                self.env.sim.physics_client.removeBody(self.env.sim._bodies_idx['goal_finger_1'])
                self.env.sim.physics_client.removeBody(self.env.sim._bodies_idx['goal_finger_2']) 
            
        obs: Dict[str, np.ndarray] = {"success": success, 
                                      "position": position, 
                                      "velocity": velocity,
                                    #   "desired_goal": self.env_dict["desired_goal"],
                                    #   "achieved_goal": self.env_dict["achieved_goal"],
                                      "images": rendered_images,
                                      "reward": self.reward}
    

        if self.render_goal:
            if self.render:
                for camera in self.camera_names:
                    cv2.imshow(camera, display_images[camera])
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    exit() # kill on q press
        else:
            if self.render:
                for camera in self.camera_names:
                    cv2.imshow(camera, rendered_images[camera])
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    exit() # kill on q press
        
        return obs
    
    def reset(self) -> Dict[str, np.ndarray]:
        # Reset the enviroment, making sure the that the achieved goal and desired goal aren't the same
        while True:
            self.env_dict, _ = self.env.reset()
            
            # print('env_dict reset', self.env_dict)
            # Set the EE to a random pose.
            if self.random_start:
                new_ee_position: np.ndarray = np.random.uniform(low=0.5*MIN_POSITION[:3], high=0.5*MAX_POSITION[:3])
                self.set_end_effector_position(new_ee_position)
                self.env_dict["observation"][0:3] = new_ee_position

            # Force block to be on the ground. Makes the task planar. Used if task is push
            if self.task_name == "push" and self.env_dict["desired_goal"][2] != 0.02:
                self.env_dict["desired_goal"][2] = 0.02
                self.env.sim.set_base_pose("target", self.env_dict["desired_goal"], np.array([0.0, 0.0, 0.0, 1.0]))
                self.env.task.goal = self.env_dict["desired_goal"]

            # Check to make sure the task hasn't already been completed.
            if self.task_name in ["push", "pick_and_place"]:
                if np.linalg.norm(self.env_dict["achieved_goal"] - self.env_dict["desired_goal"]) > 0.05:
                    break

            if self.task_name == "stack":
                # The panda gym stack task initiates the enviroment with the block in the air (to avoid colision), which then falls onto the table. This puts the block on the table to begin with. 
                self.env_dict["achieved_goal"][5] = 0.02
                self.env.sim.set_base_pose("object2", self.env_dict["achieved_goal"][3:], np.array([0.0, 0.0, 0.0, 1.0])) 

                # Also need to check for colision between blocks
                if np.linalg.norm(self.env_dict["achieved_goal"][:3] - self.env_dict["achieved_goal"][3:]) < 0.05:
                    continue

                # Verify the task hasn't already been completed.
                if np.linalg.norm(self.env_dict["achieved_goal"][:3] - self.env_dict["desired_goal"][:3]) > 0.05 \
                    or np.linalg.norm(self.env_dict["achieved_goal"][3:] - self.env_dict["desired_goal"][3:]) > 0.05:
                    break


        return self.get_obs()
    
    def step(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        self.env_dict, self.reward, _, _, _ = self.env.step(action)
        # print('env_dict step', self.env_dict)
        return self.get_obs()
        
    
    def step_pos(self, goal_pose: np.ndarray) -> Dict[str, np.ndarray]:
        # Take a step using the position controller
        action: np.ndarray = self.get_action(goal_pose)
        return self.step(action)
        
    def get_action(self, goal_pose: np.ndarray) -> np.ndarray:
        # Get the action from the position controller
        curr_position: np.ndarray = self.env_dict["observation"][:3]
        position_action: np.ndarray = self.K_POS*(goal_pose[:3] - curr_position)
        if len(goal_pose) == 3:
            return position_action
            
        elif len(goal_pose) == 4:
            gripper_width: float = self.env_dict["observation"][6]
            gripper_action: float = self.K_GRIP*(goal_pose[3] - gripper_width)
            return np.concatenate((position_action, [gripper_action]))

        else:
            raise NotImplementedError(f"Goal pose: {goal_pose} is not implemented yet.")
        
    def normalize_pos(self, pose: np.ndarray) -> np.ndarray:
        return (pose - MIN_POSITION)/(MAX_POSITION - MIN_POSITION)
    
    def unnormalize_pos(self, pose: np.ndarray) -> np.ndarray:
        return pose*(MAX_POSITION - MIN_POSITION) + MIN_POSITION
    
    def step_normalized_pos(self, goal_pose: np.ndarray) -> Dict[str, np.ndarray]:
        # Take a step using the normalized position controller
        goal_pose = self.unnormalize_pos(goal_pose)
        return self.step_pos(goal_pose)
    
    def normalize_grip(self, pose: np.ndarray) -> np.ndarray:
        out_pose = np.copy(pose)
        out_pose[3] = (out_pose[3] - MIN_POSITION[3])/(MAX_POSITION[3] - MIN_POSITION[3])
        return out_pose
    
    def unnormalize_grip(self, pose: np.ndarray) -> np.ndarray:
        out_pose = np.copy(pose)
        out_pose[3] = out_pose[3]*(MAX_POSITION[3] - MIN_POSITION[3]) + MIN_POSITION[3]
        return out_pose
    
    def step_normalized_grip(self, goal_pose: np.ndarray) -> Dict[str, np.ndarray]:
        # Take a step using the normalized position controller
        goal_pose = self.unnormalize_grip(goal_pose)
        return self.step_pos(goal_pose)
    
    def set_end_effector_position(self, position: np.ndarray, orientation: np.ndarray=np.array([1, 0, 0, 0])) -> None:
        joint_angles = self.env.robot.inverse_kinematics(self.env.robot.ee_link, position, orientation)
        self.env.robot.set_joint_angles(joint_angles)