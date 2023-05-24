import os
import math
import pathlib

import numpy as np
import matplotlib.pyplot as plt

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
from interfaces.cvmp_interface import CvmpInterface
from interfaces.mpc_interface import MpcInterface
from interfaces.dwa_interface import DwaInterface

import utils_test
from main_pre import Plotter

from typing import List, Tuple, Union



ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

class MainBase:

    HUMAN_SIZE = 0.2
    HUMAN_VMAX = 1.5
    HUMAN_STAGGER = 0.5
    
    ### Situation 1
    HUMAN_STARTS = [np.array([160, 20]),
                    np.array([160, 20])]  # XXX Sim world coords
    HUMAN_PATHS = [[1, 2, 9, 10, 14],
                   [8, 32, 16]]

    # ROBOT_START_POINT = np.array([160, 210, math.pi/2]) # XXX Sim world coords
    # ROBOT_PATH = [13, 12, 11, 32, 10, 9, 8]
    ROBOT_START_POINT = np.array([160, 220, math.pi/2]) # XXX Sim world coords
    ROBOT_PATH = [12, 11, 32, 10, 9, 8]

    ### Situation 2
    # HUMAN_STARTS = [np.array([240, 122])]  # XXX Sim world coords
    # HUMAN_PATHS = [[32, 11, 12]]

    # ROBOT_START_POINT = np.array([160, 165, math.pi/2]) # XXX Sim world coords
    # ROBOT_PATH = [12, 11, 32, 16]

    ### Situation 0
    # HUMAN_STARTS = [np.array([110, 20])]  # XXX Sim world coords
    # HUMAN_PATHS = [[1, 8, 9]]

    # ROBOT_START_POINT = np.array([160, 75, math.pi/2]) # XXX Sim world coords
    # ROBOT_PATH = [9, 8, 1]

    def __init__(self, max_run_time_step:int=120, cfg_fname:str='global_setting_warehouse.yaml') -> None:
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
        
        self.map_extent = (self.sim_config.corner_coords[0], 
                           self.sim_config.corner_coords[0] + self.sim_config.sim_width*self.sim_config.scale2real,
                           self.sim_config.corner_coords[1], 
                           self.sim_config.corner_coords[1] + self.sim_config.sim_height*self.sim_config.scale2real)

        self._load_map()

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
    
    def _prepare_plotter(self, plot_graph:bool=True):
        self.plotter = Plotter(self.config_mpc.ts, self.config_mpc.N_hor)
        self.plotter.prepare_plots(self.occ_map, map_extent=self.map_extent)
        if plot_graph:
            self.plotter.plot_netgraph(self.net_graph)
    
    def _prepare_interfaces(self, robot:Robot) -> Tuple[MmpInterface, KfmpInterface, MpcInterface, DwaInterface]:
        """Prepare (initialize) interfaces for all methods."""
        mmp_interface = MmpInterface(self.sim_config.mmp_cfg)
        kfmp_interface = KfmpInterface(self.sim_config.mpc_cfg, Q=np.eye(4), R=1.5*np.eye(2))
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
    
    def run_cv_prediction(self, interface:CvmpInterface, human_list:List[Human]):
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

    def run_one_step(self, robot:Robot, human_list:List[Human],
                     tracker_interface : Union[MpcInterface, DwaInterface],
                     predictor_interface:Union[KfmpInterface, MmpInterface]=None,
                     verbose:bool=False):
        """Run one step of simulation."""

        ### Motion prediction
        if predictor_interface is None:
            dyn_obs_list = [human.state.tolist() for human in human_list]
            mu_list_list = None
            std_list_list = None
            hypos_clusters_list = None
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
        if isinstance(tracker_interface, MpcInterface):
            actions, pred_states, cost, the_obs_list, current_refs = tracker_interface.run_step('work', dyn_obs_list, map_updated=True)
            action = actions[0]
            others = [current_refs]
        elif isinstance(tracker_interface, DwaInterface):
            the_obs_list = None
            action, pred_states, cost, all_traj, ok_traj, ok_cost = tracker_interface.run_step('work', dyn_obs_list)
            others = [all_traj, ok_traj, ok_cost]
        ### Scale back to sim
        robot.one_step(action=action) # NOTE actually robot
        for the_human in human_list:
            the_human.run_step(self.HUMAN_VMAX)

        if verbose:
            prt_goal   = f'Goal: {robot.path[-1]};'
            prt_action = f'Actions:({round(action[0], 4)}, {round(action[1], 4)});'
            prt_state  = f'Robot state: R/T {[round(x,4) for x in robot.state]}/{[round(x,4) for x in tracker_interface.state]};'
            prt_cost   = f'Cost:{round(cost,4)}.'
            print(prt_goal, prt_action, prt_state, prt_cost)

        return action, pred_states, cost, mu_list_list, std_list_list, hypos_clusters_list, the_obs_list, others

    def run_template(self, robot: Robot, human_list: List[Human],
                     tracker_interface: Union[MpcInterface, DwaInterface],
                     predictor_interface:Union[KfmpInterface, MmpInterface]=None):
        
        for kt in range(self.max_run_time_step):
            print(f'\rTime step: {kt}/{self.max_run_time_step};    ', end='')
            (action, pred_states, cost, 
             mu_list_list, std_list_list, hypos_clusters_list, the_obs_list, others) = self.run_one_step(robot, human_list, tracker_interface, predictor_interface)

            self.plotter.update_plot(kt, action, cost, tracker_interface.base_speed, mu_list_list, std_list_list, the_obs_list)
            self.plotter.plot_agent(robot, color='r')
            color_list = ['b', 'g', 'c', 'm', 'y']
            for i, human in enumerate(human_list):
                self.plotter.plot_agent(human, color=color_list[i])
            if isinstance(tracker_interface, MpcInterface):
                self.plotter.plot_references_mpc(tracker_interface.ref_path, tracker_interface.ref_traj, pred_states, *others)
            elif isinstance(tracker_interface, DwaInterface):
                self.plotter.plot_references_dwa(tracker_interface.ref_path, pred_states, *others)

            
            from matplotlib.lines import Line2D
            ### Situation 1
            robot_obj = Line2D([0], [0], color='red', linestyle='None', markersize=10, marker='o', label='Robot')
            robot_line = Line2D([0], [0], color='red', linewidth=2, linestyle='dotted', label='Robot trajectory')
            human_a_obj = Line2D([0], [0], color='blue', linestyle='None', markersize=8, marker='o', label='Human A')
            human_a_line = Line2D([0], [0], color='blue', linewidth=2, linestyle='dotted', label='Human A trajectory')
            human_b_obj = Line2D([0], [0], color='green', linestyle='None', markersize=8, marker='o', label='Human B')
            human_b_line = Line2D([0], [0], color='green', linewidth=2, linestyle='dotted', label='Human B trajectory')

            path_obj = Line2D([0], [0], color='red', linestyle='None', markersize=8, marker='x', label='Path point')
            path_line = Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Reference path')
            obs_obj = Line2D([0], [0], color='red', linestyle='None', markersize=8, marker='s', markerfacecolor='none', label='Static obstacle')
            dyn_obs_obj = Line2D([0], [0], color='yellow', linestyle='None', markersize=8, marker='o', alpha=0.5, label='Predicted obstacle')
            plt.legend(handles=[robot_obj, robot_line, 
                                human_a_obj, human_a_line,
                                human_b_obj, human_b_line,
                                path_obj, path_line,
                                obs_obj, dyn_obs_obj], loc='upper right')
            ### Situation 2
            # robot_obj = Line2D([0], [0], color='red', linestyle='None', markersize=10, marker='o', label='Robot')
            # robot_line = Line2D([0], [0], color='red', linewidth=2, linestyle='dotted', label='Robot trajectory')
            # human_a_obj = Line2D([0], [0], color='blue', linestyle='None', markersize=8, marker='o', label='Human A')
            # human_a_line = Line2D([0], [0], color='blue', linewidth=2, linestyle='dotted', label='Human A trajectory')

            # path_obj = Line2D([0], [0], color='red', linestyle='None', markersize=8, marker='x', label='Path point')
            # selected_line = Line2D([0], [0], color='m', linewidth=2, linestyle='dotted', label='Selected trajectory')
            # candidate_line = Line2D([0], [0], color='m', linewidth=2, linestyle='-', label='Candidate trajectory')
            # bad_line = Line2D([0], [0], color='c', linewidth=2, linestyle='-', label='Bad trajectory')
            # plt.legend(handles=[robot_obj, robot_line, 
            #                     human_a_obj, human_a_line,
            #                     path_obj, selected_line, candidate_line, bad_line], loc='upper right')


            plt.draw()
            plt.pause(self.config_robot.ts)
            while not plt.waitforbuttonpress():  # XXX press a button to continue
                pass

            complete = tracker_interface.traj_tracker.check_termination_condition(robot.state, action, robot.path[-1])
            if complete:
                print('Complete!')
                break
        
        plt.show()

    def run(self, tracker_type:str, predictor_type:str=None, plot_graph=False):
        tracker_type = tracker_type.lower()
        predictor_type = predictor_type.lower() if predictor_type is not None else None
        if tracker_type not in ['mpc', 'dwa']:
            raise ValueError('Tracker type is not supported.')
        if predictor_type not in ['kfmp', 'mmp', None]:
            raise ValueError('Predictor type is not supported.')
        
        self._prepare_plotter(plot_graph=plot_graph)
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
        self.run_template(robot, human_list, tracker_intf, predictor_intf)



