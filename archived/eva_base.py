import os
import math
import pathlib
import warnings
import timeit

import numpy as np

from basic_map.map_tf import ScaleOffsetReverseTransform
from basic_agent import Human, Robot

from pkg_motion_prediction.data_handle.dataset import io, ImageStackDataset

from configs import WarehouseSimConfiguration
from configs import MpcConfiguration
from configs import CircularRobotSpecification
from configs import WtaNetConfiguration

from interfaces.map_interface import MapInterface
from interfaces.mmp_interface import MmpInterface
from interfaces.kfmp_interface import KfmpInterface
from interfaces.mpc_interface import MpcInterface
from interfaces.dwa_interface import DwaInterface

import utils_test

from eva_pre import *

from typing import List, Tuple, Union


warnings.filterwarnings('ignore')

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

class EvaluationBase:

    HUMAN_SIZE = 0.2
    HUMAN_VMAX = 1.5
    HUMAN_STAGGER = 0.5
    
    ### Situation 1
    # HUMAN_STARTS = [np.array([110, 20]),
    #                 np.array([160, 20])]  # XXX Sim world coords
    # HUMAN_PATHS = [[1, 2, 9, 10, 14],
    #                [8, 32, 16]]

    # ROBOT_START_POINT = np.array([160, 210, math.pi/2]) # XXX Sim world coords
    # ROBOT_PATH = [13, 12, 11, 32, 10, 9, 8]

    ### Situation 2
    HUMAN_STARTS = [np.array([240, 122])]  # XXX Sim world coords
    HUMAN_PATHS = [[32, 11, 12]]

    ROBOT_START_POINT = np.array([160, 165, math.pi/2]) # XXX Sim world coords
    ROBOT_PATH = [12, 11, 32, 16]

    ### Situation 0
    # HUMAN_STARTS = [np.array([110, 20])]  # XXX Sim world coords
    # HUMAN_PATHS = [[1, 8, 9]]

    # ROBOT_START_POINT = np.array([160, 75, math.pi/2]) # XXX Sim world coords
    # ROBOT_PATH = [9, 8, 1]

    def __init__(self, max_num_run:int=50, max_run_time_step:int=120, cfg_fname:str='global_setting_warehouse.yaml') -> None:
        self.max_num_run = max_num_run
        self.max_run_time_step = max_run_time_step

        ### Global load configuration
        self.sim_config = WarehouseSimConfiguration.from_yaml(os.path.join(ROOT_DIR, 'config', cfg_fname))
        self.config_robot = CircularRobotSpecification.from_yaml(os.path.join(ROOT_DIR, 'config', self.sim_config.mpc_cfg))
        self.config_mpc = MpcConfiguration.from_yaml(os.path.join(ROOT_DIR, 'config', self.sim_config.mpc_cfg))
        self.config_wta = WtaNetConfiguration.from_yaml(os.path.join(ROOT_DIR, 'config', self.sim_config.mmp_cfg), with_partition=True)

        ### Load ref map
        ref_map_path = os.path.join(ROOT_DIR, 'data', self.sim_config.map_dir, 'label.png')
        self.ref_map = ImageStackDataset.togray(io.imread(ref_map_path))

        ### Coodinate transformation to real world (ROS world)
        self.ct2real = ScaleOffsetReverseTransform(scale=self.sim_config.scale2real, 
                                                   offsetx_after=self.sim_config.corner_coords[0], offsety_after=self.sim_config.corner_coords[1], 
                                                   y_reverse=~self.sim_config.image_axis, y_max_before=self.sim_config.sim_height)
        
        self._load_map()
        self._load_metrics()

    def _load_metrics(self):
        self.collision_results = []
        self.smoothness_results = []
        self.clearance_results = []
        self.clearance_dyn_results = []
        self.deviation_results = []
        self.solve_time_list = []

    def _load_map(self):
        map_interface = MapInterface(self.sim_config.map_dir)
        self.occ_map = map_interface.get_occ_map_from_pgm(self.sim_config.map_file, 120, inversed_pixel=True)
        self.geo_map = map_interface.cvt_occ2geo(self.occ_map, inflate_margin=self.config_robot.vehicle_width+self.config_robot.vehicle_margin)
        self.geo_map.coords_cvt(self.ct2real) # rescale for MPC
        self.net_graph = map_interface.get_graph_from_json(self.sim_config.graph_file)

    def _prepare_agents(self) -> Tuple[Robot, List[Human]]:
        """Prepare (initialize) the simulation environment.
        
        Returns:
            robot: Robot object
            human_list: List of Human objects        
        """
        robot_start = self.ct2real(self.ROBOT_START_POINT)
        human_starts = [self.ct2real(h_s) for h_s in self.HUMAN_STARTS]

        robot_path:List[tuple] = [self.ct2real(x) for x in self.net_graph.return_given_nodelist(self.ROBOT_PATH)]
        human_path_nx_list = [self.net_graph.return_given_nodelist(h_p) for h_p in self.HUMAN_PATHS]
        human_path_list:List[List[tuple]] = [[(self.ct2real(x)) for x in path_nx] for path_nx in human_path_nx_list]

        ### Load robot and human
        robot = Robot(state=robot_start, ts=self.config_robot.ts, radius=self.config_robot.vehicle_width/2,)
        robot.set_path(robot_path)

        human_list = [Human(h_s, self.config_robot.ts, radius=self.HUMAN_SIZE, stagger=self.HUMAN_STAGGER) for h_s in human_starts]
        for the_human, the_path in zip(human_list, human_path_list):
            the_human.set_path(the_path)
        return robot, human_list
    
    def _prepare_interfaces(self, robot:Robot) -> Tuple[MmpInterface, KfmpInterface, MpcInterface, DwaInterface]:
        """Prepare (initialize) interfaces for all methods."""
        mmp_interface = MmpInterface(self.sim_config.mmp_cfg)
        kfmp_interface = KfmpInterface(self.sim_config.mpc_cfg, Q=np.eye(4), R=np.eye(2))
        mpc_interface = MpcInterface(self.sim_config.mpc_cfg, robot.state, self.geo_map, False)
        dwa_interface = DwaInterface(self.sim_config.dwa_cfg, robot.state, self.geo_map, False)

        mpc_interface.update_global_path(robot.path)
        dwa_interface.update_global_path(robot.path)
        return mmp_interface, kfmp_interface, mpc_interface, dwa_interface
    
    def run_wta_prediction(self, interface:MmpInterface, human_list:List[Human]):
        """Run WTA prediction for all humans in the list.
        
        Returns:
            mu_list_list: List of mu for all humans.
            std_list_list: List of std for all humans.
            hypos_list_all: List of hypotheses for all humans.

        Comments:
            The length of all lists is equal to the horizon length.
        """
        past_traj_NN = [self.ct2real(x.tolist(), False) for x in human_list[0].past_traj]
        hypos_list_all = interface.get_motion_prediction(past_traj_NN, self.ref_map, self.config_mpc.N_hor, self.sim_config.scale2nn, batch_size=5)
        for human in human_list[1:]:
            past_traj_NN = [self.ct2real(x.tolist(), False) for x in human.past_traj]
            hypos_list = interface.get_motion_prediction(past_traj_NN, self.ref_map, self.config_mpc.N_hor, self.sim_config.scale2nn, batch_size=5)
            hypos_list_all = [np.concatenate((x,y), axis=0) for x,y in zip(hypos_list_all, hypos_list)]
        hypos_list_all = [self.ct2real.cvt_coords(x[:,0], x[:,1]) for x in hypos_list_all] # cvt2real
        ### CGF
        hypos_clusters_list = [] # len=pred_offset
        mu_list_list  = []
        std_list_list = []
        for i in range(self.config_mpc.N_hor):
            hyposM = hypos_list_all[i]
            hypos_clusters    = utils_test.fit_DBSCAN(hyposM, eps=1, min_sample=2) # DBSCAN
            mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters, enlarge=2, extra_margin=0) # Gaussian fitting
            hypos_clusters_list.append(hypos_clusters)
            mu_list_list.append(mu_list)
            std_list_list.append(std_list)
        return mu_list_list, std_list_list, hypos_clusters_list

    def run_kf_prediction(self, interface:KfmpInterface, human_list:List[Human]):
        """Run KF prediction for all humans in the list.
        
        Returns:
            mu_list_list: List of mu for all humans.
            std_list_list: List of std for all humans.

        Comments:
            The length of all lists is equal to the horizon length.
        """
        past_traj_kf = [x.tolist() for x in human_list[0].past_traj]
        mu_list_list, std_list_list = interface.get_motion_prediction(past_traj_kf)
        mu_list_list = [[x] for x in mu_list_list]
        std_list_list = [[x] for x in std_list_list]
        for human in human_list[1:]:
            past_traj_kf = [x.tolist() for x in human.past_traj]
            positions, uncertainty = interface.get_motion_prediction(past_traj_kf)
            for i, (pos, std) in enumerate(zip(positions, uncertainty)):
                mu_list_list[i].append(pos)
                std_list_list[i].append(std)
        return mu_list_list, std_list_list

    def run_one_step(self,  
                     robot:Robot,
                     human_list:List[Human],
                     tracker_interface : Union[MpcInterface, DwaInterface],
                     predictor_interface:Union[KfmpInterface, MmpInterface]=None,):
        """Run one step of simulation."""

        ### Motion prediction
        if predictor_interface is None:
            dyn_obs_list = [human.state.tolist() for human in human_list]
        elif isinstance(predictor_interface, KfmpInterface):
            mu_list_list, std_list_list = self.run_kf_prediction(predictor_interface, human_list)
            hypos_clusters_list = None
        elif isinstance(predictor_interface, MmpInterface):
            mu_list_list, std_list_list, hypos_clusters_list = self.run_wta_prediction(predictor_interface, human_list)
            
        else:
            raise ValueError('Predictor interface is not supported.')
        
        if predictor_interface is not None:
            if isinstance(tracker_interface, MpcInterface):
                n_obs = 0
                for mu_list in mu_list_list:
                    if len(mu_list)>n_obs:
                        n_obs = len(mu_list)
                dyn_obs_list = [[[0, 0, 0, 0, 0, 1]]*self.config_mpc.N_hor for _ in range(n_obs)]
                for Tt, (mu_list, std_list) in enumerate(zip(mu_list_list, std_list_list)):
                    for Nn, (mu, std) in enumerate(zip(mu_list, std_list)): # at each time offset
                        dyn_obs_list[Nn][Tt] = [mu[0], mu[1], std[0], std[1], 0, 1] # for each obstacle
            elif isinstance(tracker_interface, DwaInterface):
                dyn_obs_list = mu_list_list
                dyn_obs_list.insert(0, [human.state.tolist() for human in human_list])
            
        ### Run
        tracker_interface.set_current_state(robot.state) # NOTE: This is the correction of the state in trajectory generator!!!
        start_time = timeit.default_timer()
        if isinstance(tracker_interface, MpcInterface):
            actions, pred_states, cost, the_obs_list, current_refs = tracker_interface.run_step('work', dyn_obs_list, map_updated=True)
            action = actions[0]
        elif isinstance(tracker_interface, DwaInterface):
            action, pred_states, cost, all_traj, ok_traj, ok_cost = tracker_interface.run_step('work', dyn_obs_list)
        solve_time = timeit.default_timer() - start_time
        ### Scale back to sim
        robot.one_step(action=action) # NOTE actually robot
        for the_human in human_list:
            the_human.run_step(self.HUMAN_VMAX)

        ### Get metrics and check flags
        static_obstacles = self.geo_map.processed_obstacle_list
        dynamic_obstacles = [human.state[:2].tolist() for human in human_list]
        dynamic_obstacle_clearance = calc_minimal_dynamic_obstacle_distance(robot.state, dynamic_obstacles)
        collision = check_collision(robot.state, static_obstacles, dynamic_obstacles)
        if collision:
            complete = False
        else:
            complete = tracker_interface.traj_tracker.check_termination_condition(robot.state, action, robot.path[-1])

        return collision, complete, solve_time, dynamic_obstacle_clearance

    def run_once(self, 
                 num_run: int, 
                 robot:Robot,
                 human_list:List[Human],
                 tracker_interface : Union[MpcInterface, DwaInterface],
                 predictor_interface:Union[KfmpInterface, MmpInterface]=None,):
        
        dyn_clearance_temp = []
        for kt in range(self.max_run_time_step):
            print(f'\rCycle: {num_run+1}/{self.max_num_run}; Time step: {kt}/{self.max_run_time_step};    ', end='')
            collision, complete, solve_time, dynamic_obstacle_clearance = self.run_one_step(robot, human_list, tracker_interface, predictor_interface)
            self.solve_time_list.append(solve_time)
            dyn_clearance_temp.append(dynamic_obstacle_clearance)
            if collision:
                print('Collision!')
                self.collision_results.append(collision)
                break
            if complete:
                print('Complete!')
                self.collision_results.append(collision)
                break

        ### Append metrics
        if not complete:
            if (not self.collision_results) or (not collision):
                print('Timeout!')
                self.collision_results.append(True)

        if not self.collision_results[-1]:
            metric_smoothness = calc_action_smoothness(tracker_interface.traj_tracker.past_actions)
            metric_clearance = calc_minimal_obstacle_distance(robot.past_traj, self.geo_map.processed_obstacle_list)
            metric_deviation = calc_deviation_distance(ref_traj=tracker_interface.ref_traj, actual_traj=robot.past_traj)
            metric_clearance_dyn = min(dyn_clearance_temp)
            self.smoothness_results.append(metric_smoothness)
            self.clearance_results.append(metric_clearance)
            self.deviation_results.append(metric_deviation)
            self.clearance_dyn_results.append(metric_clearance_dyn)

    def run_evaluation(self, tracker_type:str, predictor_type:str=None):
        tracker_type = tracker_type.lower()
        predictor_type = predictor_type.lower() if predictor_type is not None else None
        if tracker_type not in ['mpc', 'dwa']:
            raise ValueError('Tracker type is not supported.')
        if predictor_type not in ['kfmp', 'mmp', None]:
            raise ValueError('Predictor type is not supported.')
        
        for rep in range(self.max_num_run):
            robot, human_list = self._prepare_agents()
            mmp_intf, kfmp_intf, mpc_intf, dwa_intf = self._prepare_interfaces(robot)
            if tracker_type == 'mpc':
                tracker_intf = mpc_intf
            elif tracker_type == 'dwa':
                tracker_intf = dwa_intf
            if predictor_type == 'kfmp':
                predictor_intf = kfmp_intf
            elif predictor_type == 'mmp':
                predictor_intf = mmp_intf
            elif predictor_type is None:
                predictor_intf = None
            self.run_once(rep, robot, human_list, tracker_intf, predictor_intf)

    def print_results(self):
        print('='*50)
        print('Solve time mean:', round(np.mean(np.array(self.solve_time_list[10:])), 3))
        print('Solve time max:', round(np.max(np.array(self.solve_time_list[10:])), 3))
        print('-'*50)
        # print('Collision results:', collision_results)
        print('Success rate:', (len(self.collision_results)-sum(self.collision_results))/len(self.collision_results))
        print('-'*50)
        # print('Smoothness results:', smoothness_results)
        print('Smoothness mean:', np.mean(np.array(self.smoothness_results), axis=0))
        print('-'*50)
        # print('Clearance results:', clearance_results)
        print('Clearance mean:', round(np.mean(np.array(self.clearance_results)), 3))

        # print('Clearance results (dyn):', clearance_dyn_results)
        print('Clearance mean (dyn):', round(np.mean(np.array(self.clearance_dyn_results)), 3))
        print('-'*50)
        # print('Deviation results:', deviation_results)
        print('Deviation mean:', round(np.mean(np.array(self.deviation_results)), 3))
        print('Deviation max:', round(np.max(np.array(self.deviation_results)), 3) if len(self.deviation_results)>0 else np.inf)
        print('='*50)

