import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

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
    def __init__(self, inflate_margin, with_stc_obs=True):
        self.boundary_coords = [(0, 0), (16, 0), (16, 10), (0, 10)]  # in counter-clockwise ordering
        if with_stc_obs:
            self.obstacle_list = [[(0, 1.5), (0, 1.6), (9, 1.6), (9, 1.5)],
                                [(0, 8.4), (0, 8.5), (9, 8.5), (9, 8.4)],
                                [(11, 1.5), (11, 1.6), (16, 1.6), (16, 1.5)],
                                [(11, 8.4), (11, 8.5), (16, 8.5), (16, 8.4)],
                                [(3, 3.3), (3, 3.7), (4, 3.7), (4, 3.3)]] # in clock-wise ordering
        else:
            self.obstacle_list = [[(0, 1.5), (0, 1.6), (9, 1.6), (9, 1.5)],
                                [(0, 8.4), (0, 8.5), (9, 8.5), (9, 8.4)],
                                [(11, 1.5), (11, 1.6), (16, 1.6), (16, 1.5)],
                                [(11, 8.4), (11, 8.5), (16, 8.5), (16, 8.4)]] # in clock-wise ordering
        self.crossing_area = [(9, 1.5), (11, 1.5), (11, 8.5), (9, 8.5)]

        self.inflation(inflate_margin=inflate_margin)

    def __call__(self, inflated:bool=True):
        if inflated:
            return self.processed_boundary_coords, self.processed_obstacle_list
        return self.boundary_coords, self.obstacle_list
        
    def plot_map(self, ax, start=None, end=None):
        boundary = np.array(self.boundary_coords + [self.boundary_coords[0]])
        ax.plot(boundary[:,0], boundary[:,1], 'k')
        ax.plot([0, 16], [5, 5], c='orange', linestyle='--')
        ax.fill_between([0, 16], [1.6, 1.6], [8.4, 8.4], color='lightgray')
        crossing = patches.Polygon(self.crossing_area, hatch='-', fc='white', ec='gray')
        ax.add_patch(crossing)
        for obs in self.obstacle_list:
            obs_edge = obs + [obs[0]]
            xs, ys = zip(*obs_edge)
            ax.plot(xs,ys,'k')

            obs = np.array(obs)
            poly = patches.Polygon(obs, color='k')
            ax.add_patch(poly)
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
    plt.show()