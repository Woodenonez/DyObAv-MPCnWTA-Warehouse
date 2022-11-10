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
CONFIG_FN1 = 'mpc_default.yaml'
CONFIG_FN2 = 'mpc_hardcnst.yaml'
CONFIG_FN3 = 'mpc_softcnst.yaml'
# CONFIG_FN4 = 'mpc_fullcnst.yaml'
case_index = 3 # if None, give the hints
LEGEND = ['Default', 'HC only', 'SC only']

### Load configuration
config_list = []
for CONFIG_FN in [CONFIG_FN1, CONFIG_FN2, CONFIG_FN3]:
    yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    config_list.append(Configurator(yaml_fp))

### Load simulator NOTE
sim = Simulator(index=case_index, inflate_margin=(config_list[0].vehicle_width+config_list[0].vehicle_margin))

### Global path
gpp = GloablPathPlanner(external_path=Path([Node(x) for x in sim.waypoints]))
gpp.set_start_node(Node(sim.start[0], sim.start[1], sim.start[2], 0)) # must set the start point

start = node2state(gpp.start_node)
end   = node2state(gpp.final_node) # special case, otherwise iterate waypoints

### Local path
lpp = LocalPathPlanner(sim.graph)
ref_path = lpp.get_ref_path(start.tolist(), end.tolist())
ref_traj = lpp.get_ref_traj(config_list[0].ts, config_list[0].high_speed * config_list[0].lin_vel_max, start.tolist())

### Start & run MPC
traj_gen_list = []
xx_list = []
xy_list = []
uv_list = []
uomega_list = []
cost_list_list = []
solve_time_list = []
for config in config_list:
    traj_gen_list.append(TrajectoryGenerator(config, build=False, use_tcp=False, verbose=False))
    traj_gen_list[-1].set_obstacle_weights(1e4, 1e4)
    state_list, action_list, cost_list, solve_time = traj_gen_list[-1].run(ref_traj, start, end, map_manager=sim.graph, obstacle_scanner=sim.scanner, plot_in_loop=False)

    xx, xy     = np.array(state_list)[:,0],  np.array(state_list)[:,1]
    uv, uomega = np.array(action_list)[:,0], np.array(action_list)[:,1]
    xx_list.append(xx)
    xy_list.append(xy)
    uv_list.append(uv)
    uomega_list.append(uomega)
    cost_list_list.append(cost_list)
    solve_time_list.append(solve_time)


### Plot results (press any key to continue in dynamic mode if stuck)
fig = plt.figure(constrained_layout=True)
color_list = ['k', 'r', 'g']
axes = utils_plot.prepare_plot(fig, sim.graph, start=start, end=end, legend_style='compare', double_map=True, color_list=color_list, legend_list=LEGEND)
for xx,xy,uv,uomega,cost_list,c in zip(xx_list, xy_list, uv_list, uomega_list, cost_list_list, color_list):
    utils_plot.update_plot(axes, config.ts, xx, xy, uv, uomega, cost_list, color=c)
plt.show()