import os
import math
import pathlib
import warnings
import timeit

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from basic_map.map_tf import ScaleOffsetReverseTransform
from basic_agent import Human, Robot

from pkg_motion_prediction.data_handle.dataset import io, ImageStackDataset

from configs import WarehouseSimConfiguration
from configs import MpcConfiguration
from configs import CircularRobotSpecification
from configs import WtaNetConfiguration

from interfaces.mmp_interface import MmpInterface
from interfaces.map_interface import MapInterface
from interfaces.mpc_interface import MpcInterface

import utils_test

from eva_pre import *

warnings.filterwarnings('ignore')


MAX_SIM_TIME = 120
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

### Global custom
CONFIG_FILE = 'global_setting_warehouse.yml'
HUMAN_SIZE = 0.2
HUMAN_VMAX = 1.5
HUMAN_STAGGER = 0.5

ROBOT_START_POINT = np.array([160, 210, math.pi/2]) # XXX Sim world coords

HUMAN_A_START = np.array([110, 20])  # XXX Sim world coords
HUMAN_B_START = np.array([160, 20]) # XXX Sim world coords

### Global load
sim_config = WarehouseSimConfiguration.from_yaml(os.path.join(ROOT_DIR, 'config', CONFIG_FILE))
config_mpc = MpcConfiguration.from_yaml(os.path.join(ROOT_DIR, 'config', sim_config.mpc_cfg))
config_robot = CircularRobotSpecification.from_yaml(os.path.join(ROOT_DIR, 'config', sim_config.mpc_cfg))
config_wta = WtaNetConfiguration.from_yaml(os.path.join(ROOT_DIR, 'config', sim_config.mmp_cfg), with_partition=True)

### Load red map
ref_map_path = os.path.join(ROOT_DIR, 'data', sim_config.map_dir, 'label.png')
ref_map = ImageStackDataset.togray(io.imread(ref_map_path))

### Coodinate transformation to real world (ROS world)
ct2real = ScaleOffsetReverseTransform(scale=sim_config.scale2real, 
                                      offsetx_after=sim_config.corner_coords[0], 
                                      offsety_after=sim_config.corner_coords[1], 
                                      y_reverse=~sim_config.image_axis, 
                                      y_max_before=sim_config.sim_height)

### Prepare
ROBOT_START_POINT = ct2real(ROBOT_START_POINT)
HUMAN_A_START = ct2real(HUMAN_A_START)
HUMAN_B_START = ct2real(HUMAN_B_START)

### Load map
map_interface = MapInterface(sim_config.map_dir)
occ_map = map_interface.get_occ_map_from_pgm(sim_config.map_file, 120, inversed_pixel=True)
geo_map = map_interface.cvt_occ2geo(occ_map, inflate_margin=config_robot.vehicle_width+config_robot.vehicle_margin)
net_graph = map_interface.get_graph_from_json(sim_config.graph_file)

### Rescale for MPC
geo_map_rescale = geo_map
geo_map_rescale.coords_cvt(ct2real)

### Path
robot_path = [ct2real(x) for x in net_graph.return_given_nodelist([13, 12, 11, 32, 10, 9, 8])]

path_nx_a = net_graph.return_given_nodelist([1, 2, 9, 10, 14])
path_nx_b = net_graph.return_given_nodelist([8, 32, 16])
path_nx_list = [path_nx_a, path_nx_b]
human_path_list = [[(ct2real(x)) for x in path_nx] for path_nx in path_nx_list]

### Define metrics
collision_results = []
smoothness_results = []
clearance_results = []
clearance_dyn_results = []
deviation_results = []
solve_time = []

max_rep = 50
for rep in range(max_rep):
    ### Load humans
    human_a = Human(HUMAN_A_START, config_mpc.ts, radius=HUMAN_SIZE, stagger=HUMAN_STAGGER)
    human_b = Human(HUMAN_B_START, config_mpc.ts, radius=HUMAN_SIZE, stagger=HUMAN_STAGGER)
    human_list = [human_a, human_b]

    for the_human, the_path in zip(human_list, human_path_list):
        the_human.set_path(the_path)

    ### NOTE Load motion predictor and MPC controller
    mmp_interface = MmpInterface(sim_config.mmp_cfg)
    mpc_interface = MpcInterface(sim_config.mpc_cfg, ROBOT_START_POINT, geo_map_rescale, verbose=False)

    mpc_interface.update_global_path(robot_path)

    the_robot = Robot(state=ROBOT_START_POINT, ts=config_robot.ts, radius=config_robot.vehicle_width/2,)
    the_robot.set_path(robot_path)

    dyn_clearance_temp = []
    for kt in range(MAX_SIM_TIME):
        ### Motion prediction
        past_traj_NN = [ct2real(x.tolist(), False) for x in human_list[0].past_traj]
        hypos_list_all = mmp_interface.get_motion_prediction(past_traj_NN, ref_map, config_mpc.N_hor, sim_config.scale2nn, batch_size=5)
        for human in human_list[1:]:
            past_traj_NN = [ct2real(x.tolist(), False) for x in human.past_traj]
            hypos_list = mmp_interface.get_motion_prediction(past_traj_NN, ref_map, config_mpc.N_hor, sim_config.scale2nn, batch_size=5)
            hypos_list_all = [np.concatenate((x,y), axis=0) for x,y in zip(hypos_list_all, hypos_list)]

        hypos_list_all = [ct2real.cvt_coords(x[:,0], x[:,1]) for x in hypos_list_all] # cvt2real
        ### CGF
        hypos_clusters_list = [] # len=pred_offset
        mu_list_list  = []
        std_list_list = []
        for i in range(config_mpc.N_hor):
            hyposM = hypos_list_all[i]
            hypos_clusters    = utils_test.fit_DBSCAN(hyposM, eps=1, min_sample=2) # DBSCAN
            mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters, enlarge=1, extra_margin=0) # Gaussian fitting
            hypos_clusters_list.append(hypos_clusters)
            mu_list_list.append(mu_list)
            std_list_list.append(std_list)

        ### Get dynamic obstacle list for MPC
        n_obs = 0
        for mu_list in mu_list_list:
            if len(mu_list)>n_obs:
                n_obs = len(mu_list)
        full_dyn_obs_list = [[[0, 0, 0, 0, 0, 1]]*config_mpc.N_hor for _ in range(n_obs)]
        for Tt, (mu_list, std_list) in enumerate(zip(mu_list_list, std_list_list)):
            for Nn, (mu, std) in enumerate(zip(mu_list, std_list)): # at each time offset
                full_dyn_obs_list[Nn][Tt] = [mu[0], mu[1], std[0], std[1], 0, 1] # for each obstacle

        ### Run
        print(f'\rCycle: {rep+1}/{max_rep}; Time step: {kt}/{MAX_SIM_TIME};    ', end='')
        mpc_interface.set_current_state(the_robot.state) # NOTE: This is the correction of the state in trajectory generator!!!
        start_time = timeit.default_timer()
        actions, pred_states, cost, the_obs_list, current_refs = mpc_interface.run_step('work', full_dyn_obs_list, map_updated=True)
        solve_time.append(timeit.default_timer() - start_time)
        ### Scale back to sim
        action = actions[0]
        the_robot.one_step(action=action) # NOTE actually robot

        # prt_goal   = f'Goal: {the_robot.path[-1]};'
        # prt_action = f'Actions:({round(actions[0][0], 4)}, {round(actions[0][1], 4)});'
        # prt_state  = f'Robot state: R/T {[round(x,2) for x in the_robot.state]}/{[round(x,2) for x in mpc_interface.state]};'
        # prt_cost   = f'Cost:{round(cost,4)}.'
        # print(prt_goal, prt_action, prt_state, prt_cost)

        for the_human in human_list:
            the_human.run_step(HUMAN_VMAX)

        ### Get metrics
        static_obstacles = geo_map_rescale.processed_obstacle_list
        dynamic_obstacles = [human.state[:2].tolist() for human in human_list]
        dyn_clearance_temp.append(calc_minimal_dynamic_obstacle_distance(the_robot.state, dynamic_obstacles))
        metric_a = check_collision(the_robot.state, static_obstacles, dynamic_obstacles)
        if metric_a:
            print('Collision!')
            collision_results.append(metric_a)
            break

        ### Finish-check: 
        done = mpc_interface.traj_tracker.check_termination_condition(the_robot.state, action, the_robot.path[-1])
        if done:
            print('Done!')
            break
    
    ### Append metrics
    if not done:
        if not collision_results[-1]:
            print('Timeout!')
        collision_results.append(True)
    else:
        collision_results.append(False)

    if not collision_results[-1]:
        metric_b = calc_action_smoothness(mpc_interface.traj_tracker.past_actions)
        metric_c = calc_minimal_obstacle_distance(the_robot.past_traj, static_obstacles)
        metric_d = calc_deviation_distance(ref_traj=mpc_interface.ref_traj, actual_traj=the_robot.past_traj)
        metric_e = min(dyn_clearance_temp)
        smoothness_results.append(metric_b)
        clearance_results.append(metric_c)
        deviation_results.append(metric_d)
        clearance_dyn_results.append(metric_e)

print('='*50)
print('Solve time mean:', np.mean(np.array(solve_time)))
print('-'*50)
# print('Collision results:', collision_results)
print('Collision rate:', (len(collision_results)-sum(collision_results))/len(collision_results))
print('-'*50)
# print('Smoothness results:', smoothness_results)
print('Smoothness mean:', np.mean(np.array(smoothness_results), axis=0))
print('-'*50)
# print('Clearance results:', clearance_results)
print('Clearance mean:', np.mean(np.array(clearance_results)))

# print('Clearance results (dyn):', clearance_dyn_results)
print('Clearance mean (dyn):', np.mean(np.array(clearance_dyn_results)))
print('-'*50)
# print('Deviation results:', deviation_results)
print('Deviation mean:', np.mean(np.array(deviation_results)))
print('Deviation max:', np.max(np.array(deviation_results)) if len(deviation_results)>0 else np.inf)
print('='*50)

