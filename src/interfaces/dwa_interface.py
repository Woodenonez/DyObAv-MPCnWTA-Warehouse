import os
import pathlib
import itertools

import numpy as np

from basic_motion_model.motion_model import UnicycleModel

from configs import DwaConfiguration, CircularRobotSpecification
from basic_map.map_geometric import GeometricMap
from basic_map.graph_basic import NetGraph

from pkg_dwa_tracker.trajectory_tracker import TrajectoryTracker
from pkg_dwa_tracker import utils_geo

from typing import List, Tuple, Union

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

class DwaInterface:
    def __init__(self, config_file_name: str, current_state: np.ndarray, geo_map: GeometricMap, verbose=True) -> None:
        self._prt_name = 'DWAInterface'

        config_file_path = os.path.join(ROOT_DIR, 'config', config_file_name)
        self.config_dwa = DwaConfiguration.from_yaml(config_file_path)
        self.config_robot = CircularRobotSpecification.from_yaml(config_file_path)

        self.traj_tracker = TrajectoryTracker(self.config_dwa, self.config_robot, verbose=verbose)
        self.traj_tracker.load_motion_model(UnicycleModel(self.config_robot.ts))

        self.state = current_state
        self.geo_map = geo_map

        self.prepared = False

    def set_current_state(self, current_state: np.ndarray):
        self.state = current_state
        self.traj_tracker.set_current_state(current_state)

    def update_map(self, geo_map: GeometricMap):
        self.geo_map = geo_map

    def update_global_path(self, new_global_path: List[tuple]):
        self.traj_tracker.load_init_states(self.state, np.array(new_global_path[-1]))
        self.traj_tracker.set_work_mode('work')
        self.traj_tracker.set_ref_trajectory(new_global_path)
        self.ref_path = new_global_path
        self.ref_traj = self.traj_tracker.ref_traj
        self.base_speed = self.traj_tracker.base_speed
        self.prepared = True

    def run_step(self, mode, dyn_obstacle_list:Union[List[tuple], List[List[tuple]]]=None, map_updated=None) -> Tuple[np.ndarray, np.ndarray, float, List[np.ndarray], List[np.ndarray]]:
        """Run one step of MPC.
        
        Returns:
            action: The action to be executed.
            pred_states: List of predicted states.
            cost: Cost of the current step.
        """
        if not self.prepared:
            raise ValueError('MPCInterface is not prepared. Call update_global_path() first.')

        static_obstacls = self.geo_map.processed_obstacle_list
        action, self.pred_states, cost, all_trajectories, ok_trajectories, ok_cost = self.traj_tracker.run_step(self.ref_path, static_obstacls, dyn_obstacle_list, mode=mode)
        self.state = self.traj_tracker.state
        return action, self.pred_states, cost, all_trajectories, ok_trajectories, ok_cost

