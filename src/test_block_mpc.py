import os
import math
import pathlib

import numpy as np

from mpc_planner.util.config import Configurator

from mpc_planner.path_advisor.global_path_plan import GloablPathPlanner
from mpc_planner.path_advisor.local_path_plan import LocalPathPlanner
from mpc_planner.trajectory_generator import TrajectoryGenerator

from mpc_planner.scenario_simulator import Simulator
from mpc_planner.util import utils_plot
from util.basic_objclass import *

'''
File info:
    Date    - [May. 01, 2022] -> [??. ??, 2022]
    Ref     - [Trajectory generation for mobile robotsin a dynamic environment using nonlinear model predictive control, CASE2021]
            - [https://github.com/wljungbergh/mpc-trajectory-generator]
    Exe     - [Yes]
File description:
    The main file for running the trajectory planner
Comments:
    GPP - GlobalPathPlanner; 
    LPP - LocalPathPlanner; 
    TG  - TrajGenrator; 
    OS  - ObstacleScanner; 
    MPC - MpcModule
                                                                                        V [MPC] V
    [GPP] --global path & static obstacles--> [LPP] --refernece path & tunnel width--> [TG(Config)] <--dynamic obstacles-- [OS]
    GPP is assumed given, which takes info of the "map" and "static obstacles" and controller by a [Scheduler].
Branches:
    [main]: Using TCP/IP interface
    [direct_interface]: Using Python binding direct interface to Rust
    [new_demo]: Under construction
'''

def node2state(node:Node) -> NumpyState:
    return np.array([node.x, node.y, node.theta])

### Customize
# CONFIG_FN = 'mpc_test.yaml'
CONFIG_FN = 'mpc_default.yaml'
# CONFIG_FN = 'mpc_fullcnst.yaml'
# CONFIG_FN = 'mpc_hardcnst.yaml'
# CONFIG_FN = 'mpc_softcnst.yaml'
INIT_BUILD = True
PLOT_INLOOP = True
show_animation = False
save_animation = False
case_index = 3 # if None, give the hints

### Load configuration
yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
config = Configurator(yaml_fp)

### Load simulator NOTE
sim = Simulator(index=case_index, inflate_margin=(config.vehicle_width+config.vehicle_margin))

### Global path
gpp = GloablPathPlanner(external_path=Path([Node(x) for x in sim.waypoints]))
gpp.set_start_node(Node(sim.start[0], sim.start[1], sim.start[2], 0)) # must set the start point

start = node2state(gpp.start_node)
end   = node2state(gpp.final_node) # special case, otherwise iterate waypoints

### Local path
lpp = LocalPathPlanner(sim.graph)
ref_path = lpp.get_ref_path(start.tolist(), end.tolist())
ref_traj = lpp.get_ref_traj(config.ts, config.high_speed * config.lin_vel_max, start.tolist())

### Start & run MPC
traj_gen = TrajectoryGenerator(config, build=INIT_BUILD, use_tcp=False, verbose=True)
traj_gen.set_obstacle_weights(1e4, 1e4)
state_list, action_list, cost_list, solve_time = traj_gen.run(ref_traj, start, end, map_manager=sim.graph, obstacle_scanner=sim.scanner, plot_in_loop=PLOT_INLOOP)

### Plot results (press any key to continue in dynamic mode if stuck)
xx, xy     = np.array(state_list)[:,0],  np.array(state_list)[:,1]
uv, uomega = np.array(action_list)[:,0], np.array(action_list)[:,1]
utils_plot.plot_results(sim.graph, config.ts, xx, xy, uv, uomega, cost_list, start, end, animation=show_animation, scanner=sim.scanner, video=save_animation)
