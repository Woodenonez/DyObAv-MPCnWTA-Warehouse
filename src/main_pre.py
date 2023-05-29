import math
import statistics
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from shapely.geometry import Polygon, Point

from basic_map.map_occupancy import OccupancyMap
from basic_map.graph_basic import NetGraph
from basic_agent import MovingAgent

import utils_test

from typing import List, Tuple

HUMAN_SIZE = 0.2

def check_collision(state: np.ndarray, 
                    static_obstacles: List[List[tuple]], 
                    dynamic_obstacle: List[tuple]) -> bool:
    pos = Point(state[0], state[1])
    ### Check static obstacles
    for obstacle in static_obstacles:
        if Polygon(obstacle).contains(pos):
            return True
    ### Check dynamic obstacles
    for obstacle in dynamic_obstacle:
        if pos.distance(Point(obstacle[0], obstacle[1])) <= HUMAN_SIZE:
            return True
    return False

def calc_action_smoothness(action_list: List[np.ndarray]):
    speeds = np.array(action_list)[:, 0]
    angular_speeds = np.array(action_list)[:, 1]
    return [statistics.mean(np.abs(np.diff(speeds, n=2))), statistics.mean(np.abs(np.diff(angular_speeds, n=2)))]

def calc_minimal_obstacle_distance(trajectory: List[tuple], obstacles: List[List[tuple]]):
    dist_list = []
    for pos in trajectory:
        dist_list.append(min([Polygon(obs).distance(Point(pos)) for obs in obstacles]))
    return min(dist_list)

def calc_minimal_dynamic_obstacle_distance(state: np.ndarray, obstacles: List[tuple]):
    dist_list = [np.linalg.norm(state[:2]-np.array(obstacle)) for obstacle in obstacles]
    return min(dist_list)

def calc_deviation_distance(ref_traj: List[tuple], actual_traj: List[tuple]):
    deviation_dists = []
    for pos in actual_traj:
        deviation_dists.append(min([math.hypot(ref_pos[0]-pos[0], ref_pos[1]-pos[1]) for ref_pos in ref_traj]))
    return [statistics.mean(deviation_dists), max(deviation_dists)]


class Plotter:
    def __init__(self, ts: float, horizon: int):
        self.ts = ts
        self.N_hor = horizon
        self.occ_map = None

    def plot_netgraph(self, net_graph: NetGraph):
        fig1, ax1 = plt.subplots()
        if self.occ_map is not None:
            ax1.imshow(self.occ_map(), cmap='Greys')
        net_graph.plot_netgraph(ax1)
        plt.show()

    def prepare_plots(self, occ_map: OccupancyMap, map_extent: list):
        self.occ_map = occ_map
        self.map_extent = map_extent

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 4, figure=fig)
        vel_ax = fig.add_subplot(gs[0, :2])
        vel_ax.set_ylabel('Velocity [m/s]', fontsize=15)
        omega_ax = fig.add_subplot(gs[1, :2])
        omega_ax.set_ylabel('Angular velocity [rad/s]', fontsize=15)
        cost_ax = fig.add_subplot(gs[2, :2])
        cost_ax.set_xlabel('Time [s]', fontsize=15)
        cost_ax.set_ylabel('Cost', fontsize=15)
        map_ax = fig.add_subplot(gs[:, 2:])
        map_ax.set_xlabel('X [m]', fontsize=15)
        map_ax.set_ylabel('Y [m]', fontsize=15)
        
        self.fig = fig
        self.gs = gs
        self.vel_ax = vel_ax
        self.omega_ax = omega_ax
        self.cost_ax = cost_ax
        self.map_ax = map_ax

        self.vel_list   = []
        self.omega_list = []
        self.cost_list  = []

    def update_plot(self, kt: int, action: np.ndarray, cost: float, base_speed:float, mu_list_list:list=None, std_list_list:list=None, plot_obs_list:list=None):
        [ax.cla() for ax in [self.vel_ax, self.omega_ax, self.cost_ax]]
        self.vel_list.append(action[0])
        self.omega_list.append(action[1])
        self.cost_list.append(cost)

        self.vel_ax.plot([0, (kt+1)*self.ts], [base_speed, base_speed], 'r--')
        self.vel_ax.plot(np.linspace(0, self.ts*(len(self.vel_list)), len(self.vel_list)), self.vel_list, '-o', markersize = 4, linewidth=2, color='b')
        self.omega_ax.plot(np.linspace(0, self.ts*(len(self.omega_list)), len(self.omega_list)), self.omega_list, '-o', markersize = 4, linewidth=2, color='b')
        self.cost_ax.plot(np.linspace(0, self.ts*(len(self.cost_list)), len(self.cost_list)), self.cost_list, '-o', markersize = 4, linewidth=2, color='b')

        self.map_ax.cla()
        self.map_ax.set_title(f'Time: {kt*self.ts:.2f}s / {kt:.0f}')
        self.map_ax.imshow(self.occ_map(), cmap='Greys', extent=self.map_extent)

        if mu_list_list is not None:
            for i in range(self.N_hor):
                # for j in range(len(hypos_clusters_list[i])):
                #     ax.plot(hypos_clusters_list[i][j][:,0], hypos_clusters_list[i][j][:,1], 'c.')
                mu_list  = mu_list_list[i]
                std_list = std_list_list[i]
                utils_test.plot_Gaussian_ellipses(self.map_ax, mu_list, std_list, alpha=0.2)

        if plot_obs_list is not None:
            for obs in plot_obs_list:
                obs_ = obs + [obs[0]]
                self.map_ax.plot(np.array(obs_)[:,0], np.array(obs_)[:,1], 'r-', linewidth=3)

    def plot_agent(self, agent: MovingAgent, color: str):
        agent.plot_agent(self.map_ax, color=color)
        self.map_ax.plot(np.array(agent.past_traj)[:,0], np.array(agent.past_traj)[:,1],'.', color=color)

    def plot_references_mpc(self, ref_path, ref_traj, pred_states, current_refs):
        self.map_ax.plot(np.array(ref_path)[:,0],np.array(ref_path)[:,1],'rx')
        self.map_ax.plot(np.array(ref_traj)[:,0],np.array(ref_traj)[:,1],'r--')
        self.map_ax.plot(np.array(pred_states)[:,0], np.array(pred_states)[:,1], 'm.')
        self.map_ax.plot(current_refs[:, 0], current_refs[:, 1], 'gx')

    def plot_references_dwa(self, ref_path, pred_states, all_traj, ok_traj, ok_cost):
        self.map_ax.plot(np.array(ref_path)[:,0],np.array(ref_path)[:,1],'rx') # or the_robot.path
        for tr in all_traj:
            self.map_ax.plot(np.array(tr)[:,0], np.array(tr)[:,1], 'c-', linewidth=1)
        for tr, c in zip(ok_traj, ok_cost):
            self.map_ax.plot(np.array(tr)[:,0], np.array(tr)[:,1], 'm-', linewidth=1)
            self.map_ax.text(tr[-1][0], tr[-1][1], f'{round(c,2)}', fontsize=8, color='m')
        self.map_ax.plot(np.array(pred_states)[:,0], np.array(pred_states)[:,1], 'm.')



