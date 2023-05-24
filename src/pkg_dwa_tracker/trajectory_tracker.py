# System import
import os
import sys
import math
import timeit
# External import
import numpy as np
# Custom import 
from configs import DwaConfiguration, CircularRobotSpecification
from pkg_dwa_tracker import utils_geo
# Type hint
from typing import Callable, Tuple, Union, List, Dict, Any


class TrajectoryTracker:
    """Generate a smooth trajectory tracking based on the reference path and obstacle information.
    
    Attributes:
        config: DWA configuration.
        robot_spec: Robot specification.

    Functions:
        run_step: Run one step of trajectory tracking.

    Comments:
        The solver needs to be built before running the trajectory tracking. \n
        To run the tracker: 
            1. Load motion model and init states; 
            2. Set reference path and trajectory (and states maybe);
            3. Run step.
    """
    def __init__(self, config: DwaConfiguration, robot_specification: CircularRobotSpecification, verbose=False):
        self._prt_name = '[TrajTracker]'
        self.vb = verbose
        self.config = config
        self.robot_spec = robot_specification

        # Common used parameters from config
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor

        # Initialization
        self.idle = True
        self.set_work_mode(mode='work')
        self.vw_range = [robot_specification.lin_vel_min, robot_specification.lin_vel_max,
                        -robot_specification.ang_vel_max, robot_specification.ang_vel_max]

    def load_motion_model(self, motion_model: Callable) -> None:
        """The motion model should be `s'=f(s,a,ts)`.
        - The state needs to be [x, y, theta]
        - The action needs to be [v, w]
        """
        try:
            motion_model(np.array([0,0,0]), np.array([0,0]), 0)
        except Exception as e:
            raise TypeError(f'The motion model doesn\'t satisfy the requirement: {e}')
        self.motion_model = motion_model

    def load_init_states(self, current_state: np.ndarray, goal_state: np.ndarray):
        """Load the initial state and goal state.

        Arguments:
            current_state: Current state of the robot.
            goal_state: Goal state of the robot.

        Attributes:
            state: Current state of the robot.
            final_goal: Goal state of the robot.
            past_states: List of past states of the robot.
            past_actions: List of past actions of the robot.
            cost_timelist: List of cost values of the robot.
            solver_time_timelist: List of solver time of the robot.

        Comments:
            This function resets the `idx_ref_traj/path` to 0 and `idle` to False.
        """
        if (not isinstance(current_state, np.ndarray)) or (not isinstance(goal_state, np.ndarray)):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}/{type(goal_state)}.')
        self.state = current_state
        self.final_goal = goal_state # used to check terminal condition

        self.past_states  = []
        self.past_actions = []
        self.cost_timelist = []
        self.solver_time_timelist = []

        self.idx_ref_traj = 0 # for reference trajectory following
        self.idx_ref_path = 0 # for reference path following
        self.idle = False


    def calc_dynamic_window(self, last_v: float, last_w: float):
        """Calculate the dynamic window.
        
        Returns:
            dw: Dynamic window [v_min, v_max, yaw_rate_min, yaw_rate_max]
        """
        vw_now = [last_v - self.robot_spec.lin_acc_max*self.ts,
                  last_v + self.robot_spec.lin_acc_max*self.ts,
                  last_w - self.robot_spec.ang_acc_max*self.ts,
                  last_w + self.robot_spec.ang_acc_max*self.ts]

        dw = [max(self.vw_range[0], vw_now[0]), min(self.vw_range[1], vw_now[1]),
              max(self.vw_range[2], vw_now[2]), min(self.vw_range[3], vw_now[3])]

        return dw
    
    def pred_trajectory(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict the trajectory based on the current state and action.

        Arguments:
            state: Current state of the robot, should be [x, y, angle].
            action: Current action of the robot, should be [v, w].

        Returns:
            trajectory: Predicted trajectory of the robot.
        """
        x = state.copy()
        trajectory = state.copy()
        for _ in range(self.config.N_hor):
            x = self.motion_model(x, action, self.config.ts)
            trajectory = np.vstack((trajectory, x))
        return trajectory


    def calc_cost_goal_direction(self, trajectory: np.ndarray, goal_state: np.ndarray):
        """Calculate the cost based on the goal direction.
        """
        dx = goal_state[0] - trajectory[-1, 0]
        dy = goal_state[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost * self.config.q_goal_dir

    def calc_cost_static_obstacles(self, trajectory: np.ndarray, static_obstacles: List[List[tuple]], thre:float=0.5):
        dists_to_obs = []
        for obs in static_obstacles:
            dists = utils_geo.lineseg_dists(trajectory[:,:2], np.array(obs), np.array(obs[1:] + [obs[0]]))
            if np.min(dists) < 0.05:
                return np.inf
            dists_to_obs.append(np.min(dists))
        min_dist = np.min(dists_to_obs)
        if min_dist > thre:
            return 0.0
        return 1.0 / min_dist * self.config.q_stc_obstacle
    
    def calc_cost_dynamic_obstacles(self, trajectory: np.ndarray, dynamic_obstacles: List[tuple], thre:float=0.2):
        # thre += self.robot_spec.vehicle_width
        set1 = np.expand_dims(trajectory[:, :2], axis=1)
        set2 = np.expand_dims(np.array(dynamic_obstacles), axis=0)
        distances = np.sqrt(np.sum((set1 - set2)**2, axis=-1))
        min_distances = np.min(distances)
        if min_distances > 0.5:
            return 0.0
        if min_distances < thre:
            return np.inf
        return 1.0 / min_distances * self.config.q_dyn_obstacle

    def calc_cost_dynamic_obstacles_steps(self, trajectory: np.ndarray, dynamic_obstacles: List[List[tuple]], thre:float=0.2):
        # thre += self.robot_spec.vehicle_width
        all_step_min_distances = []
        for i, obs in enumerate(dynamic_obstacles):
            set1 = np.expand_dims(trajectory[i, :2], axis=1)
            set2 = np.expand_dims(np.array(obs), axis=0)
            distances = np.sqrt(np.sum((set1 - set2)**2, axis=-1))
            min_distances = np.min(distances) * np.sqrt(i+1)
            if min_distances < thre:
                return np.inf
            all_step_min_distances.append(min_distances)
        
        if np.min(all_step_min_distances) > 0.5:
            return 0.0
        return 1.0 / np.min(all_step_min_distances) * self.config.q_dyn_obstacle

    def calc_cost_speed(self, action: np.ndarray):
        return abs(action[0] - self.base_speed) * self.config.q_speed

    def calc_cost_ref_deviation(self, trajectory: np.ndarray, ref_traj: np.ndarray):
        """Calculate the cost based on the reference trajectory."""
        dists = utils_geo.lineseg_dists(trajectory[-1,:2], ref_traj[:-1,:2], ref_traj[1:,:2])
        return np.min(dists) * self.config.q_ref_deviation

    def calc_trajectory_cost(self, trajectory: np.ndarray, action: np.ndarray, ref_path: np.ndarray, goal_state: np.ndarray, 
                             static_obstacles: List[List[tuple]], dynamic_obstacles: Union[List[tuple], List[List[tuple]]]) -> float:
        cost_speed = self.calc_cost_speed(action)
        cost_goal_dir = self.calc_cost_goal_direction(trajectory, goal_state)
        cost_ref_deviation = self.calc_cost_ref_deviation(trajectory, ref_path)
        cost_static_obs = 0.0
        cost_dynamic_obs = 0.0
        if static_obstacles is not None:
            cost_static_obs = self.calc_cost_static_obstacles(trajectory, static_obstacles)
        if dynamic_obstacles is not None:
            if np.array(dynamic_obstacles).ndim == 2:
                cost_dynamic_obs = self.calc_cost_dynamic_obstacles(trajectory, dynamic_obstacles)
            elif np.array(dynamic_obstacles).ndim == 3:
                cost_dynamic_obs = self.calc_cost_dynamic_obstacles_steps(trajectory, dynamic_obstacles[1:]) + \
                                   self.calc_cost_dynamic_obstacles(trajectory, dynamic_obstacles[0])
            else:
                raise ValueError('Dynamic obstacles should be a list of tuples or a list of list of tuples.')
        return cost_speed + cost_goal_dir + cost_ref_deviation + cost_static_obs + cost_dynamic_obs


    def set_work_mode(self, mode:str='safe'): # change base speed and tuning parameters
        """Set the basic work mode (base speed and weight parameters) of the MPC solver.
        Arguments:
            `mode`: "aligning" (start) or "safe" (20% speed) or "work" (80% speed) or "super" (full speed)
        Attributes:
            `base_speed`: The reference speed of the robot.
        """
        ### Base/reference speed
        if mode == 'aligning':
            self.base_speed = self.robot_spec.lin_vel_max*0.1
        elif mode == 'safe':
            self.base_speed = self.robot_spec.lin_vel_max*0.2
        elif mode == 'work':
            self.base_speed = self.robot_spec.lin_vel_max*0.8
        elif mode == 'super':
            self.base_speed = self.robot_spec.lin_vel_max*1.0
        else:
            raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def set_current_state(self, current_state: np.ndarray):
        """To synchronize the current state of the robot with the MPC solver."""
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

    def set_ref_trajectory(self, ref_path: List[tuple], ref_traj:List[tuple]=None):
        """Set the global reference trajectory.

        Arguments:
            ref_path: Global reference path, must be given.
            ref_traj: Global reference trajectory, if not given, generated by the simple `get_ref_traj` method.

        Attributes:
            idx_ref_path: Index of the reference path, start from 0.
            idx_ref_traj: Index of the reference trajectory, start from 0.
            ref_path: Global reference path.
            ref_traj: Global reference trajectory.
        """
        self.idx_ref_path = 0
        self.idx_ref_traj = 0
        self.ref_path = ref_path
        if ref_traj is not None:
            self.ref_traj = ref_traj
        else:
            self.ref_traj = self.get_ref_traj(self.ts, ref_path, self.state, self.base_speed)

    def check_termination_condition(self, state: np.ndarray, action: np.ndarray, final_goal: np.ndarray) -> bool:
        if np.allclose(state[:2], final_goal[:2], atol=0.5, rtol=0) and abs(action[0]) < 0.4:
            terminated = True
            self.idle = True
            if self.vb:
                print(f"{self._prt_name} DWA solution found.")
        else:
            terminated = False
        return terminated


    @staticmethod
    def get_ref_traj(ts: float, ref_path: List[tuple], state: tuple, speed: float) -> List[tuple]:
        """Generate the global reference trajectory from the reference path.

        Returns:
            ref_traj: x, y coordinates and the heading angles
        """
        x, y = state[0], state[1]
        x_next, y_next = ref_path[0][0], ref_path[0][1]
        
        ref_traj = []
        path_idx = 0
        traveling = True
        while(traveling):# for n in range(N):
            while(True):
                dist_to_next = math.hypot(x_next-x, y_next-y)
                if dist_to_next < 1e-9:
                    path_idx += 1
                    x_next, y_next = ref_path[path_idx][0], ref_path[path_idx][1]
                    break
                x_dir = (x_next-x) / dist_to_next
                y_dir = (y_next-y) / dist_to_next
                eta = dist_to_next/speed # estimated time of arrival
                if eta > ts: # move to the target node for t
                    x = x+x_dir*speed*ts
                    y = y+y_dir*speed*ts
                    break # to append the position
                else: # move to the target node then set a new target
                    x = x+x_dir*speed*eta
                    y = y+y_dir*speed*eta
                    path_idx += 1
                    if path_idx > len(ref_path)-1 :
                        traveling = False
                        break
                    else:
                        x_next, y_next = ref_path[path_idx][0], ref_path[path_idx][1]
            if not dist_to_next < 1e-9:
                ref_traj.append((x, y, math.atan2(y_dir,x_dir)))
        return ref_traj


    def run_step(self, ref_path: List[tuple], static_obstacles: List[List[tuple]], dynamic_obstacles: Union[List[tuple], List[List[tuple]]], mode:str='work'):
        """Run the trajectory planner for one step.

        Returns:
            best_u: The best action to take.
            trajectory: The predicted trajectory.
            min_cost: The cost of the best trajectory.
        """
        self.set_work_mode(mode)
        ### Correct the reference speed ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        if dist_to_goal < self.base_speed*self.N_hor*self.ts:
            self.base_speed = min(2 * dist_to_goal / self.N_hor / self.ts, self.robot_spec.lin_vel_max)

        min_cost = float("inf")
        x_init = self.state.copy()
        best_u = np.zeros(self.nu)
        best_trajectory = self.state.reshape(1, -1)
        last_u = self.past_actions[-1] if len(self.past_actions) else np.zeros(self.nu)
        all_trajectories = []
        ok_trajectories = []
        ok_cost = []

        start_time = timeit.default_timer()
        ### Get dynamic window ###
        dw = self.calc_dynamic_window(last_u[0], last_u[1])

        ### Get reference states ###
        for v in np.arange(dw[0], dw[1], self.config.vel_resolution):
            for w in np.arange(dw[2], dw[3], self.config.ang_resolution):
                u = np.array([v, w])
                trajectory = self.pred_trajectory(x_init, u)
                all_trajectories.append(trajectory)
                cost = self.calc_trajectory_cost(trajectory, u, np.array(ref_path), self.final_goal, static_obstacles, dynamic_obstacles)
                if cost < np.inf:
                    ok_trajectories.append(trajectory)
                    ok_cost.append(cost)
                if min_cost > cost:
                    min_cost = cost
                    best_u = u
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.config.stuck_threshold:
                        best_u[1] = -self.robot_spec.ang_vel_max
        solver_time = timeit.default_timer() - start_time

        self.state = best_trajectory[0,:]
        self.past_states.append(self.state)
        self.past_actions += [best_u]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)

        return best_u, best_trajectory, min_cost, all_trajectories, ok_trajectories, ok_cost
