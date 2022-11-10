import os
import pathlib
import itertools

from mpc_planner.util.config import Configurator
from mpc_planner.path_advisor.global_path_plan import GloablPathPlanner
from mpc_planner.path_advisor.local_path_plan import LocalPathPlanner
from mpc_planner.trajectory_generator import TrajectoryGenerator
from mpc_planner.obstacle_scanner.dynamic_obstacle_scanner import ObstacleScanner
from util import utils_geo

from util.basic_datatype import *
from util.basic_objclass import *


class MpcInterface:
    def __init__(self, config_file_name:str, current_state:NumpyState, graph_for_mpc:GeometricMap, external_base_path:Path, init_build:bool=False, path_segment:bool=False) -> None:
        self.__prtname = 'MPCInterface'

        self.build = init_build
        config_file_path = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'config', config_file_name)
        self.config = Configurator(config_file_path)
        self.path_seg = path_segment

        current_node = Node(current_state[0], current_state[1], current_state[2], id=0) # dynamic node id-0
        self.update_global_path(external_base_path, current_node)
        self.update_local_path(graph_for_mpc)

        final_goal_state = np.array(self.gpp.final_node()[:3])
        next_goal_state = np.array(self.gpp.next_node()[:3])
        self.traj_gen = TrajectoryGenerator(self.config, build=self.build, verbose=True)
        if self.path_seg:
            self.traj_gen.load_init_states(current_state, next_goal_state, final_goal_state)
        else:
            self.traj_gen.load_init_states(current_state, final_goal_state, final_goal_state)

        self.idx_ref = 0 # for running step-by-step

    def update_global_path(self, new_global_path:Path, current_node:Node=None):
        if current_node is None:
            current_node = Node(self.traj_gen.state[0], self.traj_gen.state[1], self.traj_gen.state[2], id=0)
        self.gpp = GloablPathPlanner(external_path=new_global_path)
        self.gpp.set_start_node(current_node) # must set the start point

    def update_local_path(self, new_graph_map:GeometricMap=None):
        if new_graph_map is not None:
            self.lpp = LocalPathPlanner(new_graph_map)
        start = self.gpp.start_node()[:3]
        end   = self.gpp.next_node()[:3]
        if self.path_seg:
            self.ref_path = self.lpp.get_ref_path(start, end)
        else:
            self.ref_path = []
            for i in range(len(self.gpp.global_path)-1):
                start_temp = self.gpp.global_path[i][:3]
                end_temp   = self.gpp.global_path[i+1][:3]
                self.ref_path += self.lpp.get_ref_path(start_temp, end_temp)[:-1]
            self.ref_path.append(self.lpp.get_ref_path(start_temp, end_temp)[-1])
            self.lpp.ref_path = self.ref_path
        self.ref_traj = self.lpp.get_ref_traj(self.config.ts, self.config.high_speed * self.config.lin_vel_max, start)

    def move_to_next_node(self):
        if self.path_seg: # this only work if the ref path is segmented
            self.gpp.move_to_next_node()
            self.update_local_path()
            next_goal_state = np.array([self.gpp.next_node.x, self.gpp.next_node.y, 0])
            self.traj_gen.set_next_goal(next_goal_state)

    def run(self, obstacle_scanner:ObstacleScanner):
        start_np = np.array([self.gpp.start_node.x, self.gpp.start_node.y, 0])
        end_np   = np.array([self.gpp.final_node.x, self.gpp.final_node.y, 0]) # XXX special case, otherwise iterate waypoints
        state_list, action_list, cost_list, solve_time_list = self.traj_gen.run(self.ref_traj, start_np, end_np, map_manager=self.lpp.graph_map, obstacle_scanner=obstacle_scanner, plot_in_loop=False)
        return state_list, action_list, cost_list, solve_time_list

    def run_step(self, mode, full_dyn_obstacle_list:list=None, map_updated:bool=True) -> Tuple[List[NumpyAction], List[NumpyState], int, float, List[List[tuple]]]:
        # full_dyn_obstacle_list should be [[[T=1],[T=2],...], [...], ...]
        if map_updated:
            _, closest_obstacle_list = self.get_stc_constraints()
        dyn_constraints = self.get_dyn_constraints(full_dyn_obstacle_list)
        actions, self.pred_states, idx_next, cost, current_refs, base_speed = self.traj_gen.run_step(self.idx_ref, self.stc_constraints, dyn_constraints, self.ref_traj, mode=mode)
        self.idx_ref = idx_next
        return actions, self.pred_states, idx_next, cost, closest_obstacle_list, current_refs, base_speed

    def get_stc_constraints(self) -> Tuple[list, List[List[tuple]]]:
        n_stc_obs = self.config.Nstcobs * self.config.nstcobs
        self.stc_constraints = [0.0] * n_stc_obs
        map_obstacle_list = self.get_closest_n_stc_obstacles()
        for i, map_obstacle in enumerate(map_obstacle_list):
            b, a0, a1 = utils_geo.polygon_halfspace_representation(np.array(map_obstacle))
            self.stc_constraints[i*self.config.nstcobs : (i+1)*self.config.nstcobs] = (b+a0+a1)
        return self.stc_constraints, map_obstacle_list

    def get_dyn_constraints(self, full_dyn_obstacle_list=None):
        params_per_dyn_obs  = self.config.N_hor * self.config.ndynobs
        dyn_constraints = [0.0] * self.config.Ndynobs * params_per_dyn_obs
        if full_dyn_obstacle_list is not None:
            for i, dyn_obstacle in enumerate(full_dyn_obstacle_list):
                dyn_constraints[i*params_per_dyn_obs:(i+1)*params_per_dyn_obs] = list(itertools.chain(*dyn_obstacle))
        return dyn_constraints

    def get_closest_n_stc_obstacles(self) -> List[List[tuple]]:
        full_obs_list = self.lpp.graph_map.processed_obstacle_list
        short_obs_list = []
        dists_to_obs = []
        for obs in full_obs_list:
            dists = utils_geo.lineseg_dists(self.traj_gen.state[:2], np.array(obs), np.array(obs[1:] + [obs[0]]))
            dists_to_obs.append(np.min(dists))
        selected_idc = np.argpartition(dists_to_obs, self.config.Nstcobs)[:self.config.Nstcobs]
        for i in selected_idc:
            short_obs_list.append(full_obs_list[i])
        return short_obs_list

    def get_closest_n_dyn_obstacles(self, full_dyn_obstacle_list) -> List[List[tuple]]:
        pass