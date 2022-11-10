import os, sys
from typing import Tuple, List

from extremitypathfinder.extremitypathfinder import PolygonEnvironment

from util.basic_objclass import GeometricMap

import matplotlib.pyplot as plt            # XXX for debugging
from util.mapnet import plot_geometric_map # XXX for debugging

'''
File info:
    Name    - [visibility]
    Date    - [Jan. 01, 2021] -> [Aug. 20, 2021]
    Exe     - [Yes]
File description:
    [Path advisor] Generate the reference path via the visibility graph and A* algorithm.
'''

class VisibilityPathFinder:
    '''
    Description:
        Generate the reference path via the visibility graph and A* algorithm.
    Attributes:
        env  <object>         - The environment object of solving the visibility graph.
        path <list of tuples> - The shortest path from the visibility graph.
    Functions
        __prepare               <pre> - Prepare the visibility graph including preprocess the map.
        get_ref_path            <get> - Get the (shortest) refenence path.
    '''
    def __init__(self, graph_map:GeometricMap, verbose=False):
        self.__print_name = '[LocalPath-Visibility]'
        self.graph = graph_map
        self.vb = verbose
        self.__prepare()

    def __prepare(self):
        self.env = PolygonEnvironment()
        self.env.store(self.graph.processed_boundary_coords, self.graph.processed_obstacle_list) # pass obstacles and boundary to environment
        # self.env.store(self.graph.processed_boundary_coords, self.graph.processed_obstacle_list[:2])
        self.env.prepare() # prepare the visibility graph 

    def get_ref_path(self, start_pos:tuple, end_pos:tuple) -> Tuple[List[tuple], List[tuple]]:
        '''
        Description:
            Generate the initially guessed path based on obstacles and boundaries specified during preparation.
        Arguments:
            start_pos <tuple> - The x,y coordinates.
            end_pos   <tuple> - The x,y coordinates.
        Return:
            Path <list of tuples> - List of coordinates of the inital path
        '''
        if self.vb:
            print(f'{self.__print_name} Reference path generated.')

        # map_info = {'boundary': self.graph.processed_boundary_coords, 'obstacle_list':self.graph.processed_obstacle_list}
        # _, ax = plt.subplots()
        # plot_geometric_map(ax, map_info, start_pos[:2], end_pos[:2])
        # plt.show() XXX

        self.path, dist = self.env.find_shortest_path(start_pos[:2], end_pos[:2]) # 'dist' are distances of every segments.
        return self.path, dist

    
    