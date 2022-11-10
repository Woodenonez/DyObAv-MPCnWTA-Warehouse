import sys, math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

'''
File info:
    Name    - [ftd_graph]
    Exe     - [Yes]
File description:
    This contains the map of the synthetic factory traffic dataset.
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
        start <tuple> - The start point in 2D coordinates (useless here).
        end   <tuple> - The end point in 2D coordinates (useless here).
        index <int>   - Not used.
    Attributes:
        boundary_coordinates <list of tuples> - Each tuple is a vectex of the boundary polygon.
                                              - Defined in counter-clockwise ordering.
        obstacle_list        <list of lists>  - Each sub-list represents an obstacle in the form of a list of tuples.
                                              - Defined in clockwise ordering.
    Functions
        plot_map <vis> - Visualization of the map. Plot directly.
    '''
    def __init__(self, start, end, index=0):
        self.boundary_coordinates = [(0,0), (10.0,0), (10.0,2.5), (6.0,2.5), (6.0,4.5), (10.0,4.5), (10.0,6.5), (6.0,6.5), 
                                     (6.0,10.0), (4.0,10.0), (4.0,6.5), (0,6.5), (0,4.5), (4.0,4.5), (4.0,2.5), (0,2.5)]  # in counter-clockwise ordering
        self.obstacle_list = [[(5.3,2.3), (5.3,4.5), (5.7,4.5), (5.7,2.5)]] # in clock-wise ordering
        self.start = start
        self.end = end

    def plot_map(self):
        boundary = self.boundary_coordinates + [self.boundary_coordinates[0]]
        boundary = np.array(boundary)
        fig, ax = plt.subplots()
        plt.plot(boundary[:,0], boundary[:,1], 'k')
        for obs in self.obstacle_list:
            obs_edge = obs + [obs[0]]
            xs, ys = zip(*obs_edge)
            plt.plot(xs,ys,'b')

            obs = np.array(obs)
            poly = patches.Polygon(obs, color='skyblue')
            ax.add_patch(poly)
        plt.plot(self.start[0], self.start[1], 'b*')
        plt.plot(self.end[0], self.end[1], 'r*')
        plt.show()

if __name__ == '__main__':
    start = (1, 1.0, math.radians(0))
    end = (9.0, 1.0, math.radians(0))
    graph = Graph(start, end)
    graph.plot_map()