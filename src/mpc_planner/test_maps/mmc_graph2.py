import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection

import pyclipper

'''
File info:
    Name    - [mmc_graph]
    Exe     - [Yes]
File description:
    This contains the map of the synthetic multimodal crossing (mmc) data.
File content:
    Graph     <class> - Define the map.
Comments:
    None
'''

class GraphTemplate:
    def __init__(self, external_boundary:list=None, external_obstacles:list=None) -> None:
        self.boundary_coordinates = external_boundary
        self.obstacle_list = external_obstacles

    def plot_map(self, ax, start=None, end=None):
        pass

class Graph:
    '''
    Description:
        Define a map.
    Arguments:
        None
    Attributes:
        boundary_coordinates <list of tuples> - Each tuple is a vectex of the boundary polygon.
                                              - Defined in counter-clockwise ordering.
        obstacle_list        <list of lists>  - Each sub-list represents an obstacle in the form of a list of tuples.
                                              - Defined in clockwise ordering.
    Functions
        plot_map <vis> - Visualization of the map. Plot directly.
    '''
    def __init__(self, inflate_margin=0):
        self.boundary_coords = [(0, 0), (12, 0), (12, 16), (0, 16)]  # in counter-clockwise ordering
        self.obstacle_list = [[(0, 0), (0, 3), (3, 3), (3, 0)],
                              [(0, 9), (0, 12), (3, 12), (3, 9)],
                              [(9, 9), (9, 12), (12, 12), (12, 9)],
                              [(9, 0), (9, 3), (12, 3), (12, 0)]] # in clock-wise ordering
        self.sidewalk_list = [[(0, 3), (0, 4), (4, 4), (4, 0), (3, 0), (3, 3)],
                              [(0, 8), (0, 9), (3, 9), (3, 12), (4, 12), (4, 8)],
                              [(8, 8), (8, 12), (9, 12), (9, 9), (12, 9), (12, 8)],
                              [(8, 0), (8, 4), (12, 4), (12, 3), (9, 3), (9, 0)]]
        self.crossing_area = [[(4, 3), (4, 4), (8, 4), (8, 3)],
                              [(3, 4), (3, 8), (4, 8), (4, 4)],
                              [(4, 8), (4, 9), (8, 9), (8, 8)],
                              [(8, 4), (8, 8), (9, 8), (9, 4)]]

        self.inflation(inflate_margin=inflate_margin)

    def __call__(self, inflated:bool=True):
        if inflated:
            return self.processed_boundary_coords, self.processed_obstacle_list
        return self.boundary_coords, self.obstacle_list

    def plot_map(self, ax, start=None, end=None):
        boundary = np.array(self.boundary_coords + [self.boundary_coords[0]])
        # Boundary
        ax.plot(boundary[:,0], boundary[:,1], 'k')
        # Lane
        ax.plot([0, 12], [6, 6], c='orange', linestyle='--')
        ax.plot([6, 6], [0, 12], c='orange', linestyle='--')
        ax.fill_between([0, 12], [4, 4], [8, 8], color='lightgray')
        ax.fill_between([4, 8], [0, 0], [12, 12], color='lightgray')
        # Area
        for cs in self.crossing_area:
            cs_edge = cs + [cs[0]]
            xs, ys = zip(*cs_edge)
            plt.plot(xs,ys,'gray')
            poly = patches.Polygon(np.array(cs), hatch='-', color='white')
            ax.add_patch(poly)
        for sw in self.sidewalk_list:
            # sw_edge = sw + [sw[0]]
            # xs, ys = zip(*sw_edge)
            # plt.plot(xs,ys,'k')
            poly = patches.Polygon(np.array(sw), color='gray')
            ax.add_patch(poly)
        for obs in self.obstacle_list:
            obs_edge = obs + [obs[0]]
            xs, ys = zip(*obs_edge)
            plt.plot(xs,ys,'k')
            poly = patches.Polygon(np.array(obs), color='k')
            ax.add_patch(poly)
        # Start and end
        if start is not None:
            ax.plot(self.start[0], self.start[1], 'b*')
        if end is not None:
            ax.plot(self.end[0], self.end[1], 'r*')
        ax.axis('equal')

    def inflation(self, inflate_margin):
        self.inflator = pyclipper.PyclipperOffset()
        self.processed_obstacle_list   = self.__preprocess_obstacles(self.obstacle_list, 
                                                                     pyclipper.scale_to_clipper(inflate_margin))
        self.processed_boundary_coords = self.__preprocess_obstacle( pyclipper.scale_to_clipper(self.boundary_coords), 
                                                                     pyclipper.scale_to_clipper(-inflate_margin))

    def __preprocess_obstacle(self, obstacle, inflation):
        self.inflator.Clear()
        self.inflator.AddPath(obstacle, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_obstacle = pyclipper.scale_from_clipper(self.inflator.Execute(inflation))[0]
        return inflated_obstacle    
    
    def __preprocess_obstacles(self, obstacle_list, inflation):
        inflated_obstacles = []
        for obs in obstacle_list:
            obstacle = pyclipper.scale_to_clipper(obs)
            inflated_obstacle = self.__preprocess_obstacle(obstacle, inflation)
            inflated_obstacle.reverse() # obstacles are ordered clockwise
            inflated_obstacles.append(inflated_obstacle)
        return inflated_obstacles

if __name__ == '__main__':
    graph = Graph(0)
    _, ax = plt.subplots()
    graph.plot_map(ax)
    ax.text(6,-0.5,'S')
    ax.text(6,12.5,'N')
    ax.text(-0.5,6,'W')
    ax.text(12.5,6,'E')
    plt.show()