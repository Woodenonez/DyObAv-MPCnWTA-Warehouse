import os
import math
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from basic_map.map_tf import ScaleOffsetReverseTransform
from basic_agent import Human, Robot

from pkg_motion_prediction.data_handle.dataset import io, ImageStackDataset

from configs import WarehouseSimConfiguration
from configs import MpcConfiguration
from configs import CircularRobotSpecification

from interfaces.kfmp_interface import KfmpInterface
from interfaces.map_interface import MapInterface
from interfaces.mpc_interface import MpcInterface

import utils_test


MAX_SIM_TIME = 1_000
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

### Global custom
CONFIG_FILE = 'global_setting_warehouse.yaml'
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

### Load red map
ref_map_path = os.path.join(ROOT_DIR, 'data', sim_config.map_dir, 'label.png')
ref_map = ImageStackDataset.togray(io.imread(ref_map_path))

### Coodinate transformation to real world (ROS world)
ct2real = ScaleOffsetReverseTransform(scale=sim_config.scale2real, 
                                      offsetx_after=sim_config.corner_coords[0], 
                                      offsety_after=sim_config.corner_coords[1], 
                                      y_reverse=~sim_config.image_axis, 
                                      y_max_before=sim_config.sim_height)

map_extent = (sim_config.corner_coords[0], 
              sim_config.corner_coords[0] + sim_config.sim_width*sim_config.scale2real,
              sim_config.corner_coords[1], 
              sim_config.corner_coords[1] + sim_config.sim_height*sim_config.scale2real)

### Prepare
ROBOT_START_POINT = ct2real(ROBOT_START_POINT)
HUMAN_A_START = ct2real(HUMAN_A_START)
HUMAN_B_START = ct2real(HUMAN_B_START)

### Load map
map_interface = MapInterface(sim_config.map_dir)
occ_map = map_interface.get_occ_map_from_pgm(sim_config.map_file, 120, inversed_pixel=True)
geo_map = map_interface.cvt_occ2geo(occ_map, inflate_margin=config_robot.vehicle_width+config_robot.vehicle_margin)
net_graph = map_interface.get_graph_from_json(sim_config.graph_file)

### Load humans
human_a = Human(HUMAN_A_START, config_mpc.ts, radius=HUMAN_SIZE, stagger=HUMAN_STAGGER)
human_b = Human(HUMAN_B_START, config_mpc.ts, radius=HUMAN_SIZE, stagger=HUMAN_STAGGER)
human_list = [human_a, human_b]

path_nx_a = net_graph.return_given_nodelist([1, 2, 9, 10, 14])
path_nx_b = net_graph.return_given_nodelist([8, 32, 16])
path_nx_list = [path_nx_a, path_nx_b]

path_list = [[(ct2real(x)) for x in path_nx] for path_nx in path_nx_list]
for the_human, the_path in zip(human_list, path_list):
    the_human.set_path(the_path)

### Run and Vis
fig1, ax1 = plt.subplots()
ax1.imshow(occ_map(), cmap='Greys')
net_graph.plot_netgraph(ax1)

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

base_path_nx = net_graph.return_given_nodelist([13, 12, 11, 32, 10, 9, 8])
base_path = [ct2real(x) for x in base_path_nx]

### Rescale for MPC
geo_map_rescale = geo_map
geo_map_rescale.coords_cvt(ct2real)

### NOTE Load motion predictor and MPC controller
kfmp_interface = KfmpInterface(sim_config.mpc_cfg, Q=np.eye(4), R=np.eye(2))
mpc_interface = MpcInterface(sim_config.mpc_cfg, ROBOT_START_POINT, geo_map_rescale)

mpc_interface.update_global_path(base_path)

the_robot = Robot(state=ROBOT_START_POINT, ts=config_robot.ts, radius=config_robot.vehicle_width/2)
the_robot.set_path(base_path)

vel_list   = []
omega_list = []
cost_list  = []
for kt in range(MAX_SIM_TIME):
    ### Plan (re-plan the path or generate local reference states)
    pass

    ### Motion prediction
    past_traj_kf = [x.tolist() for x in human_list[0].past_traj]
    mu_list_list, std_list_list = kfmp_interface.get_motion_prediction(past_traj_kf)
    mu_list_list = [[x] for x in mu_list_list]
    std_list_list = [[x] for x in std_list_list]
    for human in human_list[1:]:
        past_traj_kf = [x.tolist() for x in human.past_traj]
        positions, uncertainty = kfmp_interface.get_motion_prediction(past_traj_kf)
        for i, (pos, std) in enumerate(zip(positions, uncertainty)):
            mu_list_list[i].append(pos)
            std_list_list[i].append(std)

    ### Get dynamic obstacle list for MPC
    n_obs = 0
    for mu_list in mu_list_list:
        if len(mu_list)>n_obs:
            n_obs = len(mu_list)
    full_dyn_obs_list = [[[0, 0, 0, 0, 0, 1]]*config_mpc.N_hor for _ in range(n_obs)]
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
    mpc_interface.set_current_state(the_robot.state) # NOTE: This is the correction of the state in trajectory generator!!!
    actions, pred_states, cost, the_obs_list, current_refs = mpc_interface.run_step('work', full_dyn_obs_list, map_updated=True)
    ### Scale back to sim
    action = actions[0]
    the_robot.one_step(action=action) # NOTE actually robot

    prt_action = f'Actions:({round(actions[0][0], 4)}, {round(actions[0][1], 4)});'
    prt_state  = f'Robot state: R/T {[round(x,4) for x in the_robot.state]}/{[round(x,4) for x in mpc_interface.state]};'
    prt_cost   = f'Cost:{round(cost,4)}.'
    print(prt_action, prt_state, prt_cost)

    for the_human in human_list:
        the_human.run_step(HUMAN_VMAX)

    ### Vis
    vel_ax.cla()
    omega_ax.cla()
    cost_ax.cla()

    vel_list.append(action[0])
    omega_list.append(action[1])
    cost_list.append(cost)
    vel_ax.plot([0, (kt+1)*config_mpc.ts], [mpc_interface.base_speed, mpc_interface.base_speed], 'r--')
    vel_ax.plot(np.linspace(0, config_mpc.ts*(len(vel_list)), len(vel_list)), vel_list, '-o', markersize = 4, linewidth=2, color='b')
    omega_ax.plot(np.linspace(0, config_mpc.ts*(len(omega_list)), len(omega_list)), omega_list, '-o', markersize = 4, linewidth=2, color='b')
    cost_ax.plot(np.linspace(0, config_mpc.ts*(len(cost_list)), len(cost_list)), cost_list, '-o', markersize = 4, linewidth=2, color='b')

    ax.cla()
    ax.set_title(f'Time: {kt*config_mpc.ts:.2f}s / {kt:.0f}')
    ax.imshow(occ_map(), cmap='Greys', extent=map_extent)

    # color_list = ['b', 'r', 'g', 'y', 'c'] * 4 # XXX
    # color_list[0] = 'k' # XXX
    for i in range(config_mpc.N_hor):
        # for j in range(len(hypos_clusters_list[i])):
        #     ax.plot(hypos_clusters_list[i][j][:,0], hypos_clusters_list[i][j][:,1], 'c.')
        mu_list  = mu_list_list[i]
        std_list = std_list_list[i]
        utils_test.plot_Gaussian_ellipses(ax, mu_list, std_list, alpha=0.2)

    for obs in the_obs_list:
        obs_ = obs + [obs[0]]
        ax.plot(np.array(obs_)[:,0], np.array(obs_)[:,1], 'r-', linewidth=3)

    the_robot.plot_agent(ax, color='r')
    human_a.plot_agent(ax, color='b')
    human_b.plot_agent(ax, color='g')
    ax.plot(np.array(human_a.past_traj)[:,0], np.array(human_a.past_traj)[:,1],'b.')
    ax.plot(np.array(human_b.past_traj)[:,0], np.array(human_b.past_traj)[:,1],'g.')

    ax.plot(np.array(the_robot.past_traj)[:,0], np.array(the_robot.past_traj)[:,1],'r.')
    ax.plot(np.array(mpc_interface.ref_path)[:,0],np.array(mpc_interface.ref_path)[:,1],'rx') # or the_robot.path
    ax.plot(np.array(mpc_interface.ref_traj)[:,0],np.array(mpc_interface.ref_traj)[:,1],'r--')
    ax.plot(np.array(pred_states)[:,0], np.array(pred_states)[:,1], 'm.')
    ax.plot(current_refs[:, 0], current_refs[:, 1], 'gx')

    plt.draw()
    plt.pause(config_mpc.ts)
    while not plt.waitforbuttonpress():  # XXX press a button to continue
        pass

    ### Finish-check: If humans finish their paths, assign new paths. If the robot reach the current goal, move to next (if run path_segment).
    # robot_pos = (the_robot.state[0], the_robot.state[1])
    # goal_pos  = (traj_generator.gpp.next_node.x, traj_generator.gpp.next_node.y)
    # if np.hypot(robot_pos[0]-goal_pos[0], robot_pos[1]-goal_pos[1]) < 50:
    #     traj_generator.move_to_next_node()

plt.show()

