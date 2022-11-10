import os
import sys
import copy
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpc_planner.util import utils_plot

from main_pre import prepare_map, prepare_params
from motion_prediction.mmp_interface import MmpInterface
from motion_prediction.util import utils_test
from mpc_planner.mpc_interface import MpcInterface

from util import mapnet
from util import basic_agent
from util.basic_objclass import *

MAX_SIM_TIME = 1_000

### Global custom
CONFIG_FILE = 'global_setting_warehouse.yml'

ROBOT_START_POINT = np.array([160, 210, math.pi/2]) # XXX Sim world coords

HUMAN_A_START = np.array([110, 20])  # XXX Sim world coords
HUMAN_B_START = np.array([160, 20]) # XXX Sim world coords

### Global load
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
params = prepare_params(CONFIG_FILE, ROOT_DIR)
SCENE   = params['SCENE']
MMP_CFG = params['MMP_CFG']
MPC_CFG = params['MPC_CFG']
STIME   = params['STIME']
PRED_OFFSET = params['PRED_OFFSET']

SCALE2NN    = params['SCALE2NN']
SCALE2REAL  = params['SCALE2REAL']
IMAGE_AXIS  = params['IMAGE_AXIS']
CORNER_COORDS = params['CORNER_COORDS']
SIM_WIDTH  = params['SIM_WIDTH']
SIM_HEIGHT = params['SIM_HEIGHT']

ROBOT_SIZE = 0.5

HUMAN_SIZE = 0.2
HUMAN_VMAX = 1.5
HUMAN_STAGGER = 0.5

### Coodinate transformation to real world (ROS world)
ct2real = CoordTransform(scale=SCALE2REAL, offsetx_after=CORNER_COORDS[0], offsety_after=CORNER_COORDS[1], 
                         y_reverse=~IMAGE_AXIS, y_max_before=SIM_HEIGHT)
map_extent = (CORNER_COORDS[0], CORNER_COORDS[0]+SIM_WIDTH*SCALE2REAL,
              CORNER_COORDS[1], CORNER_COORDS[1]+SIM_HEIGHT*SCALE2REAL)

### Prepare
ROBOT_START_POINT = ct2real(ROBOT_START_POINT)
HUMAN_A_START = ct2real(HUMAN_A_START)
HUMAN_B_START = ct2real(HUMAN_B_START)

the_map, ref_map = prepare_map(SCENE, ROOT_DIR, inversed_pixel=True)

### Load scene, humans, robot, and planner (just for local simulation)
map_info = {'map_image':the_map, 'threshold':120}
scene_graph = mapnet.SceneGraph(scene=SCENE, map_type='occupancy', map_info=map_info)
the_planner = basic_agent.Planner(scene_graph.NG)
human_a = basic_agent.Human(HUMAN_A_START, radius=HUMAN_SIZE, stagger=HUMAN_STAGGER)
human_b = basic_agent.Human(HUMAN_B_START, radius=HUMAN_SIZE, stagger=HUMAN_STAGGER)
human_list = [human_a, human_b]
# human_list = [human_a]
# human_list = [human_b]

path_nx_a = scene_graph.NG.return_given_path([1, 2, 9, 10, 14])
path_nx_b = scene_graph.NG.return_given_path([8, 32, 16])
path_nx_list = [path_nx_a, path_nx_b]
# path_nx_list = [path_nx_a]
# path_nx_list = [path_nx_b]

path_list = [Path([Node(list(ct2real(np.array(x)))) for x in path_nx()]) for path_nx in path_nx_list]
for the_human, the_path in zip(human_list, path_list):
    the_human.set_path(the_path)

### Run and Vis
fig1, ax1 = plt.subplots()
scene_graph.plot_netgraph(ax1)

fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 4, figure=fig)
vel_ax = fig.add_subplot(gs[0, :2])
vel_ax.set_ylabel('Velocity [m/s]', fontsize=15)
omega_ax = fig.add_subplot(gs[1, :2])
omega_ax.set_ylabel('Angular velocity [rad/s]', fontsize=15)
cost_ax = fig.add_subplot(gs[2, :2])
cost_ax.set_xlabel('Time [s]', fontsize=15)
cost_ax.set_ylabel('Cost', fontsize=15)
ax = fig.add_subplot(gs[:, 2:])
ax.set_xlabel('X [m]', fontsize=15)
ax.set_ylabel('Y [m]', fontsize=15)

_, backup_paths_nx = the_planner.k_shortest_paths(source=13, target=8, k=1)
base_path_nx = backup_paths_nx[0]
base_path = Path([Node(list(ct2real(np.array(x)))) for x in base_path_nx()])
# base_path = Path([Node(1,0,0), Node(1,8,0), Node(8,8,0)]) # XXX

### Rescale for MPC
geo_map_rescale = scene_graph.base_map.get_geometric_map()
geo_map_rescale.coords_cvt(ct2real)
geo_map_rescale.inflation(ROBOT_SIZE)

### NOTE Load motion predictor and MPC controller
motion_predictor = MmpInterface(MMP_CFG)
traj_generator = MpcInterface(MPC_CFG, ROBOT_START_POINT, geo_map_rescale, external_base_path=base_path, init_build=False)

the_robot = basic_agent.Robot(state=ROBOT_START_POINT, radius=ROBOT_SIZE)
the_robot.set_path(base_path)

map_updated = True
hypos_list_all = [] # len=pred_offset
vel_list   = []
omega_list = []
cost_list  = []
for kt in range(MAX_SIM_TIME):
    ### Plan
    pass

    ### Motion prediction
    past_traj_NN = Trajectory([Node(list(ct2real(np.array(x),False))) for x in human_list[0].past_traj()])
    hypos_list_all = motion_predictor.get_motion_prediction(past_traj_NN, ref_map, PRED_OFFSET, SCALE2NN, batch_size=5)
    for human in human_list[1:]:
        past_traj_NN = Trajectory([Node(list(ct2real(np.array(x),False))) for x in human.past_traj()])
        hypos_list = motion_predictor.get_motion_prediction(past_traj_NN, ref_map, PRED_OFFSET, SCALE2NN, batch_size=5)
        hypos_list_all = [np.concatenate((x,y), axis=0) for x,y in zip(hypos_list_all, hypos_list)]

    hypos_list_all = [ct2real.cvt_coords(x[:,0], x[:,1]) for x in hypos_list_all] # cvt2real
    ### CGF
    hypos_clusters_list = [] # len=pred_offset
    mu_list_list  = []
    std_list_list = []
    for i in range(PRED_OFFSET):
        hyposM = hypos_list_all[i]
        hypos_clusters    = utils_test.fit_DBSCAN(hyposM, eps=1, min_sample=2) # DBSCAN
        mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters, enlarge=2, extra_margin=0*ROBOT_SIZE/2) # Gaussian fitting
        hypos_clusters_list.append(hypos_clusters)
        mu_list_list.append(mu_list)
        std_list_list.append(std_list)

    ### Get dynamic obstacle list for MPC
    n_obs = 0
    for mu_list in mu_list_list:
        if len(mu_list)>n_obs:
            n_obs = len(mu_list)
    full_dyn_obs_list = [[[0, 0, 0, 0, 0, 1]]*traj_generator.config.N_hor for _ in range(n_obs)]
    for Tt, (mu_list, std_list) in enumerate(zip(mu_list_list, std_list_list)):
        for Nn, (mu, std) in enumerate(zip(mu_list, std_list)): # at each time offset
            full_dyn_obs_list[Nn][Tt] = [mu[0], mu[1], std[0], std[1], 0, 1] # for each obstacle
        # XXX
        # if Nn+1 < n_obs:
        #     for i in range((Nn+1),n_obs):
        #         full_dyn_obs_list[i][Tt] = [mu[0], mu[1], std[0], std[1], 0, 1] # for each obstacle
        # XXX
    # full_dyn_obs_list = None # XXX

    ### Run
    print(f'Current time step: {kt}')
    traj_generator.traj_gen.set_current_state(the_robot.state) # NOTE: This is the correction of the state in trajectory generator!!!
    actions, pred_states, _, cost, the_obs_list, current_refs, base_speed = traj_generator.run_step('work', full_dyn_obs_list, map_updated)
    ### Scale back to sim
    action = actions[0]
    the_robot.one_step(ts=STIME, action=action) # NOTE actually robot
    map_updated = True # if you want to keep updating the static obstacles, then true

    prt_action = f'Actions:({round(actions[0][0], 4)}, {round(actions[0][1], 4)});'
    prt_state  = f'Robot state: R/T {[round(x,4) for x in the_robot.state]}/{[round(x,4) for x in traj_generator.traj_gen.state]};'
    prt_cost   = f'Cost:{round(cost,4)}.'
    print(prt_action, prt_state, prt_cost)

    for the_human in human_list:
        the_human.run_step(STIME, HUMAN_VMAX)

    ### Vis
    vel_ax.cla()
    omega_ax.cla()
    cost_ax.cla()

    vel_list.append(action[0])
    omega_list.append(action[1])
    cost_list.append(cost)
    vel_ax.plot([0, (kt+1)*STIME], [base_speed, base_speed], 'r--')
    utils_plot.plot_action(vel_ax, vel_list, STIME)
    utils_plot.plot_action(omega_ax, omega_list, STIME)
    utils_plot.plot_action(cost_ax, cost_list, STIME)

    ax.cla()
    ax.imshow(the_map, cmap='Greys', extent=map_extent)

    # color_list = ['b', 'r', 'g', 'y', 'c'] * 4 # XXX
    # color_list[0] = 'k' # XXX
    for i in range(PRED_OFFSET):
        # for j in range(len(hypos_clusters_list[i])):
        #     ax.plot(hypos_clusters_list[i][j][:,0], hypos_clusters_list[i][j][:,1], 'c.')
        mu_list  = mu_list_list[i]
        std_list = std_list_list[i]
        utils_test.plot_Gaussian_ellipses(ax, mu_list, std_list, alpha=0.2)

    ax.plot(traj_generator.gpp.next_node.x, traj_generator.gpp.next_node.y, 'ro')
    for obs in the_obs_list:
        obs_ = obs + [obs[0]]
        ax.plot(np.array(obs_)[:,0], np.array(obs_)[:,1], 'r-', linewidth=3)

    the_robot.plot_agent(ax, color='r')
    human_a.plot_agent(ax, color='b')
    human_b.plot_agent(ax, color='g')
    ax.plot(np.array(human_a.past_traj())[:,0], np.array(human_a.past_traj())[:,1],'b.')
    ax.plot(np.array(human_b.past_traj())[:,0], np.array(human_b.past_traj())[:,1],'g.')

    ax.plot(np.array(the_robot.past_traj())[:,0], np.array(the_robot.past_traj())[:,1],'r.')
    ax.plot(np.array(traj_generator.ref_path)[:,0],np.array(traj_generator.ref_path)[:,1],'rx') # or the_robot.path
    ax.plot(np.array(traj_generator.ref_traj)[:,0],np.array(traj_generator.ref_traj)[:,1],'r--')
    ax.plot(np.array(pred_states)[:,0], np.array(pred_states)[:,1], 'm.')
    ax.plot(current_refs[0::3], current_refs[1::3], 'gx')

    plt.draw()
    plt.pause(STIME)
    while not plt.waitforbuttonpress():  # XXX press a button to continue
        pass

    ### Finish-check: If humans finish their paths, assign new paths. If the robot reach the current goal, move to next (if run path_segment).
    robot_pos = (the_robot.state[0], the_robot.state[1])
    goal_pos  = (traj_generator.gpp.next_node.x, traj_generator.gpp.next_node.y)
    if np.hypot(robot_pos[0]-goal_pos[0], robot_pos[1]-goal_pos[1]) < 50:
        traj_generator.move_to_next_node()

    for the_human in human_list:
        if the_human.get_next_goal(STIME, HUMAN_VMAX) is None:
            new_path_nx = scene_graph.NG.return_random_path(start_node_index=the_human.path[-1][2], num_traversed_nodes=3)
            new_path = Path([Node(list(ct2real(np.array(x)))) for x in new_path_nx()])
            the_human.set_path(new_path)

plt.show()

