import os, sys
import math

from typing import Union, List

### Import specific path-finding algorithm here
from planner_mpc.path_advisor.planner_visibility import VisibilityPathFinder
from util.basic_objclass import OccupancyMap

from util.basic_objclass import GeometricMap, OccupancyMap
from util.basic_datatype import *

'''
File info:
    None
File description:
    (What does this file do?)
File content (important ones):
    ClassA      <class> - (Basic usage).
    ClassB      <class> - (Basic usage).
    function_A  <func>  - (Basic usage).
    function_B  <func>  - (Basic usage).
Comments:
    (Things worthy of attention.)
'''

class LocalPathPlanner:
    def __init__(self, graph_map:Union[GeometricMap, OccupancyMap], verbose=False):
        # The local planner should take global path and map as inputs
        self.graph_map = graph_map
        self.path_planner = VisibilityPathFinder(graph_map=graph_map, verbose=verbose)

    def get_ref_path(self, start:State, end:State) -> List[tuple]:
        self.ref_path = self.path_planner.get_ref_path(start, end)
        if isinstance(self.ref_path, tuple):
            self.ref_path = self.ref_path[0] # if there are multiple outputs, then the first one must be the path
        return self.ref_path

    def get_ref_traj(self, ts:SamplingTime, reference_speed:float, current_state:State) -> List[State]:
        '''
        Description:
            Generate the reference trajectory from the reference path.
        Return:
            ref_traj <list> - List of x, y coordinates and the heading angles
        '''
        x, y = current_state[0], current_state[1]
        x_next, y_next = self.ref_path[0][0], self.ref_path[0][1]
        
        ref_traj = []
        path_idx = 0
        traveling = True
        while(traveling):# for n in range(N):
            while(True):
                dist_to_next = math.hypot(x_next-x, y_next-y)
                if dist_to_next < 1e-9:
                    path_idx += 1
                    x_next, y_next = self.ref_path[path_idx][0], self.ref_path[path_idx][1]
                    break
                x_dir = (x_next-x) / dist_to_next
                y_dir = (y_next-y) / dist_to_next
                eta = dist_to_next/reference_speed # estimated time of arrival
                if eta > ts: # move to the target node for t
                    x = x+x_dir*reference_speed*ts
                    y = y+y_dir*reference_speed*ts
                    break # to append the position
                else: # move to the target node then set a new target
                    x = x+x_dir*reference_speed*eta
                    y = y+y_dir*reference_speed*eta
                    path_idx += 1
                    if path_idx > len(self.ref_path)-1 :
                        traveling = False
                        break
                    else:
                        x_next, y_next = self.ref_path[path_idx][0], self.ref_path[path_idx][1]
            if not dist_to_next < 1e-9:
                ref_traj.append((x, y, math.atan2(y_dir,x_dir)))
        return ref_traj
        
