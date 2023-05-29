import copy
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from basic_motion_model.motion_model import OmnidirectionalModel
from basic_motion_model.motion_model import UnicycleModel

from matplotlib.axes import Axes
from typing import List, Union, Callable


class MovingAgent():
    def __init__(self, state:np.ndarray, ts: float, radius:float=1.0, stagger:float=0.0):
        """Moving agent.
        
        Arguments:
            state: Initial state of the agent.
            ts: Sampling time.
            radius: Radius of the agent (assuming circular).
            stagger: Stagger of the agent.

        Attributes:
            r: Radius of the agent.
            ts: Sampling time.
            state: Current state of the agent.
            stagger: Stagger of the agent.
            motion_model: Motion model of the agent.
            past_traj: List of np.ndarray, past trajectory of the agent.
            with_path: Whether the agent has a path to follow.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f'State must be numpy.ndarry, got {type(state)}.')
        self.r = radius
        self.ts = ts
        self.state = state
        self.stagger = stagger
        self.motion_model = OmnidirectionalModel(ts)
        self.past_traj = [self.state]
        self.with_path = False

    def set_path(self, path: List[tuple]):
        self.with_path = True
        self.path = path
        self.coming_path = copy.deepcopy(path)
        self.past_traj = [self.state]

    def get_next_goal(self, vmax: float) -> Union[tuple, None]:
        if not self.with_path:
            raise RuntimeError('Path is not set yet.')
        if not self.coming_path:
            return None
        dist_to_next_goal = math.hypot(self.coming_path[0][0] - self.state[0], self.coming_path[0][1] - self.state[1])
        if dist_to_next_goal < (vmax*self.ts):
            self.coming_path.pop(0)
        if self.coming_path:
            return self.coming_path[0]
        else:
            return None

    def get_action(self, next_path_node: tuple, vmax: float) -> np.ndarray:
        stagger = random.choice([1,-1]) * random.randint(0,10)/10*self.stagger
        dist_to_next_node = math.hypot(self.coming_path[0][0] - self.state[0], self.coming_path[0][1] - self.state[1])
        dire = ((next_path_node[0] - self.state[0])/dist_to_next_node, 
                (next_path_node[1] - self.state[1])/dist_to_next_node)
        action:np.ndarray = np.array([dire[0]*vmax+stagger, dire[1]*vmax+stagger])
        return action

    def one_step(self, action: np.ndarray):
        self.state = self.motion_model(self.state, action)
        self.past_traj.append(self.state)

    def run_step(self, vmax:float) -> bool:
        next_path_node = self.get_next_goal(vmax)
        if next_path_node is None:
            return False
        action = self.get_action(next_path_node, vmax)
        self.one_step(action)
        return True

    def run(self, path: List[tuple], ts:float=.2, vmax:float=0.5):
        self.set_path(path)
        done = False
        while (not done):
            done = self.run_step(ts, vmax)

    def plot_agent(self, ax:Axes, color:str='b', ct:Callable=None):
        if ct is not None:
            robot_patch = patches.Circle(ct(self.state[:2]), self.r, color=color)
        else:
            robot_patch = patches.Circle(self.state[:2], self.r, color=color)
        ax.add_patch(robot_patch)


class Human(MovingAgent):
    def __init__(self, state: np.ndarray, ts: float, radius:int, stagger:int):
        super().__init__(state, ts, radius, stagger)


class Robot(MovingAgent):
    def __init__(self, state: np.ndarray, ts: float, radius:int):
        super().__init__(state, ts, radius, 0)
        self.motion_model = UnicycleModel(self.ts, rk4=True)
