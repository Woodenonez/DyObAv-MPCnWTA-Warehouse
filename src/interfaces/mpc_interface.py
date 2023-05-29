import os
import pathlib
import itertools

import numpy as np

from basic_motion_model.motion_model import UnicycleModel

from configs import MpcConfiguration, CircularRobotSpecification
from basic_map.map_geometric import GeometricMap
from basic_map.graph_basic import NetGraph

from pkg_mpc_tracker.trajectory_tracker import TrajectoryTracker
from pkg_mpc_tracker import utils_geo

from typing import List, Tuple

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

class MpcInterface:
    def __init__(self, config_file_name: str, current_state: np.ndarray, geo_map: GeometricMap, verbose=True) -> None:
        self._prt_name = 'MPCInterface'

        config_file_path = os.path.join(ROOT_DIR, 'config', config_file_name)
        self.config_mpc = MpcConfiguration.from_yaml(config_file_path)
        self.config_robot = CircularRobotSpecification.from_yaml(config_file_path)

        self.traj_tracker = TrajectoryTracker(self.config_mpc, self.config_robot, verbose=verbose)
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

    def run_step(self, mode, full_dyn_obstacle_list:list=None, map_updated:bool=True) -> Tuple[List[np.ndarray], List[np.ndarray], float, List[List[tuple]], np.ndarray]:
        """Run one step of MPC.
        
        Returns:
            actions: List of actions to be executed.
            pred_states: List of predicted states.
            cost: Cost of the current step.
            closest_obstacle_list: List of closest obstacles.
            current_refs: List of current reference states.
        """
        # full_dyn_obstacle_list should be [[[T=0],[T=1],[T=2],...], [...], ...]
        if not self.prepared:
            raise ValueError('MPCInterface is not prepared. Call update_global_path() first.')

        if map_updated:
            stc_constraints, closest_obstacle_list = self.get_stc_constraints()
        dyn_constraints = self.get_dyn_constraints(full_dyn_obstacle_list)
        actions, self.pred_states, current_refs, cost = self.traj_tracker.run_step(stc_constraints, dyn_constraints, mode=mode)
        self.state = self.traj_tracker.state
        return actions, self.pred_states, cost, closest_obstacle_list, current_refs

    def get_stc_constraints(self) -> Tuple[list, List[List[tuple]]]:
        n_stc_obs = self.config_mpc.Nstcobs * self.config_mpc.nstcobs
        stc_constraints = [0.0] * n_stc_obs
        map_obstacle_list = self.get_closest_n_stc_obstacles()
        for i, map_obstacle in enumerate(map_obstacle_list):
            b, a0, a1 = utils_geo.polygon_halfspace_representation(np.array(map_obstacle))
            stc_constraints[i*self.config_mpc.nstcobs : (i+1)*self.config_mpc.nstcobs] = (b+a0+a1)
        return stc_constraints, map_obstacle_list

    def get_dyn_constraints(self, full_dyn_obstacle_list=None):
        params_per_dyn_obs = (self.config_mpc.N_hor+1) * self.config_mpc.ndynobs
        dyn_constraints = [0.0] * self.config_mpc.Ndynobs * params_per_dyn_obs
        if full_dyn_obstacle_list is not None:
            for i, dyn_obstacle in enumerate(full_dyn_obstacle_list):
                dyn_constraints[i*params_per_dyn_obs:(i+1)*params_per_dyn_obs] = list(itertools.chain(*dyn_obstacle))
        return dyn_constraints

    def get_closest_n_stc_obstacles(self) -> List[List[tuple]]:
        full_obs_list = self.geo_map.processed_obstacle_list
        short_obs_list = []
        dists_to_obs = []
        for obs in full_obs_list:
            dists = utils_geo.lineseg_dists(self.state[:2], np.array(obs), np.array(obs[1:] + [obs[0]]))
            dists_to_obs.append(np.min(dists))
        selected_idc = np.argpartition(dists_to_obs, self.config_mpc.Nstcobs)[:self.config_mpc.Nstcobs]
        for i in selected_idc:
            short_obs_list.append(full_obs_list[i])
        return short_obs_list

    def get_closest_n_dyn_obstacles(self, full_dyn_obstacle_list) -> List[List[tuple]]:
        pass