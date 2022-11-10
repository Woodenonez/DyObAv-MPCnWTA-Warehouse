#%% General import
import os, sys
import pathlib
import math
import numpy as np
import matplotlib.pyplot as plt

#%% Test geometric map
from util import mapnet
from main_pre import prepare_map

ROBOT_SIZE = 0.5
SCENE = 'warehouse'
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
the_map, ref_map = prepare_map(SCENE, ROOT_DIR, inversed_pixel=True)
map_info = {'map_image':the_map, 'threshold':120}
scene_graph = mapnet.SceneGraph(scene=SCENE, map_type='occupancy', map_info=map_info)

geo_map = scene_graph.base_map.get_geometric_map(inflation=ROBOT_SIZE)
map_info = {'boundary':geo_map.boundary_coords, 'obstacle_list':geo_map.obstacle_list, 'inflation':20}
scene_map_geo = mapnet.SceneGraph(scene=SCENE, map_type='geometric', map_info=map_info) # just for plotting
fig2, [ax2_1, ax2_2] = plt.subplots(1, 2)
ax2_1.imshow(scene_graph.base_map(), cmap='Greys')
scene_map_geo.plot_map(ax2_2)
ax2_2.invert_yaxis()
ax2_2.axis('equal')
fig2.tight_layout()
plt.show()
sys.exit(0)

#%% Test datatype
from util.utils_sl import read_pgm_and_process
from util.basic_objclass import OccupancyMap, GeometricMap
### Load map
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
map_path     = os.path.join(ROOT_DIR, 'data', 'warehouse_sim_original', 'mymap.pgm') # bookstore_sim_original, warehouse_sim_original
ref_map_path = os.path.join(ROOT_DIR, 'data', 'warehouse_sim_original', 'label.png')
with open (map_path, 'rb') as pgmf:
    raw_map = read_pgm_and_process(pgmf, inversed_pixel=True)
    # raw_map[0:40, 950:980] = 0
print(raw_map.shape)

the_map = OccupancyMap(raw_map, occupancy_threshold=120)
print(the_map(binary_scale=True).shape)

edge_map = the_map.get_edge_map(dilation_size=0)
the_geo_map:GeometricMap = the_map.get_geometric_map()
boundary, obstacle_list = the_geo_map()

_, [ax, ax1] = plt.subplots(1,2)
ax.imshow(the_map(), cmap='Greys')
ax1.imshow(edge_map, cmap='Greys')
for coords in obstacle_list:
    coords += [coords[0]]
    ax.plot(np.array(coords)[:, 1], np.array(coords)[:, 0], '-r', linewidth=2)
    ax1.plot(np.array(coords)[:, 1], np.array(coords)[:, 0], '-r', linewidth=2)
plt.show()

sys.exit(0)

#%% Test agents in agent.py
from util.basic_agent import MovingAgent
from util.mapnet import SceneGraph

sgraph = SceneGraph('bookstore')

_, [ax1, ax2] = plt.subplots(1,2)
sgraph.plot_nodes(ax1, with_text=True)
sgraph.plot_edges(ax1)

path = sgraph.return_random_path(start_node_index=3, num_traversed_nodes=10)

ts = 0.1

stagger = 10
vmax = 70 # 1m = 33.3px, reasonable speed is 34~70px/s

obj = MovingAgent(path[0], stagger)
obj.run(path, ts, vmax)

# ------------------------
# ax.axis('off')
# graph.plot_map(ax)
# sgraph.plot_path(path, ax2, style='go--')
# ax2.plot(np.array(obj.traj)[:,0],np.array(obj.traj)[:,1],'.')
# ax2.set_aspect('equal', 'box')
# plt.tight_layout()
# ------------------------
# plt.show()

#%% Test planners in agent.py
import numpy as np
import matplotlib.pyplot as plt
from util.basic_agent import Planner, lineseg_dists
from util.mapnet import SceneGraph
sgraph = SceneGraph('bookstore')

planner = Planner(sgraph.NG)
weight_list = []
for e in planner.NG.edges:
    weight_list.append(planner.NG[e[0]][e[1]]['weight'])
all_edge_positions = planner.NG.get_all_edge_positions()
# print(all_edge_positions)

a = np.array([x[0] for x in all_edge_positions])
b = np.array([x[1] for x in all_edge_positions])
