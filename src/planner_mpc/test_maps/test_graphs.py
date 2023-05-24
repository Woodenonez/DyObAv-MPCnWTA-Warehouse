import sys, math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import pyclipper

'''
File info:
    Name    - [test_graphs]
    Date    - [Jan. 01, 2021] -> [Aug. 20, 2021]
    Ref     - [Trajectory generation for mobile robotsin a dynamic environment using nonlinear model predictive control, CASE2021]
            - [https://github.com/wljungbergh/mpc-trajectory-generator]
    Exe     - [Yes]
File description:
    This contains the original test maps from the reference. The way of defining the graph is restructured.
    'Graph' offers map information.
File content:
    Graph     <class> - Define the map.
    TestGraph <class> - Retrieve pre-defined maps by index.
Comments:
    None
'''

class Graph:
    '''
    Description:
        Define a map
    Arguments:
        start <tuple> - The start point in 2D coordinates (useless here).
        end   <tuple> - The end point in 2D coordinates (useless here).
        index <int>   - The index to retrieve pre-defined map.
    Attributes:
        boundary_coordinates <list of tuples> - Each tuple is a vectex of the boundary polygon.
                                              - Defined in counter-clockwise ordering.
        obstacle_list        <list of lists>  - Each sub-list represents an obstacle in the form of a list of tuples.
                                              - Defined in clockwise ordering.
    Functions
        plot_map <vis> - Visualization of the map. Plot directly.
    '''
    def __init__(self, inflate_margin, index=11):
        g = TestGraphs()
        g_dict = g.get_graph(index)
        self.boundary_coords   = g_dict['boundary_coordinates']    # in counter-clockwise ordering
        self.obstacle_list     = g_dict['obstacle_list']           # in clock-wise ordering
        self.start             = g_dict['start']           # XXX Just for testing
        self.end               = g_dict['end']             # XXX Just for testing
        self.dyn_obs_list      = g_dict['dyn_obs_list']    # XXX Just for testing

        self.inflation(inflate_margin=inflate_margin)

    def __call__(self, inflated:bool=True):
        if inflated:
            return self.processed_boundary_coords, self.processed_obstacle_list
        return self.boundary_coords, self.obstacle_list

    def plot_map(self, ax):
        boundary = self.boundary_coords + [self.boundary_coords[0]]
        boundary = np.array(boundary)
        ax.plot(boundary[:,0], boundary[:,1], 'k')
        for obs in self.obstacle_list:
            obs_edge = obs + [obs[0]]
            xs, ys = zip(*obs_edge)
            ax.plot(xs,ys,'b')

            obs = np.array(obs)
            poly = patches.Polygon(obs, color='skyblue')
            ax.add_patch(poly)
        ax.plot(self.start[0], self.start[1], 'b*')
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

class TestGraphs:
    '''
    Description:
        Store pre-defined maps.
    Attributes:
        graphs <list of dicts> - Contain all pre-defined maps in the form of dictionaries.
                               - Each dictionary has {boundary_coordinates, obstacle_list, start, end, dyn_obs_list}
    Functions
        gen_tests <pre> - Prepare all pre-defined maps.
        get_graph <get> - Get the graph of given index.
    '''
    def __init__(self):
        self.graphs = []
        self.gen_tests()

    def gen_tests(self):
        # boundary_coordinates in counter-clockwise ordering
        # obstacle (obstacle_list) in clock-wise ordering
    
        ############### 1st Graph ############################# 
        boundary_coordinates = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        obstacle_list = [ [(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)] ]
        start = (1,1, math.radians(0))
        end = (8,8, math.radians(90))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})
        
        ############### 2nd Graph ############################# 
        boundary_coordinates = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
        obstacle_list = [
            [(5.0, 0.0), (5.0, 15.0), (7.0, 15.0), (7.0, 0.0)],
            [(12.0, 12.5),(12.0,20.0),(15.0,20.0),(15.0,12.5)], 
            [(12.0, 0.0),(12.0,7.5),(15.0,7.5),(15.0, 0.0)] ]
        start = (1,5, math.radians(45))
        end = (19,10, math.radians(0))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 3rd Graph #############################
        boundary_coordinates = [(0.0, 0.0),(10.0,0.0), (10.0 ,10.0), (25.0, 10.0), (25.0,0.0), (50.0,0), (50,50), (0, 50), \
                                (0,16),(10,16),(10,45),(15,45),(15,30),(35,30),(35,15),(0,15)]
        obstacle_list = [
            [(30.0, 5.0), (30.0, 14.5), (40.0, 14.5), (40, 5.0)],
            [(45.0, 15.0),(44.0,20.0),(46.0,20.0)],
            [(25, 35),(25,40),(40, 40),(40,35)],
            [(32.0, 6.0), (32.0, 10.5), (42.0, 12.5), (42, 8.0)] ]
        start = (1,1,math.radians(225))
        end = (5,20,math.radians(270))
        # Starting point, ending point, x radius, y radius, angle
        dyn_obs_list = [
            [[17.5, 43], [22, 37.5], 0.1, 0.2, 0.5, 0.1], 
            [[40.5, 18], [37, 26], 0.1, 0.5, 0.2, 0.5],
            [[6.5, 5], [4.5, 7], 0.1, 0.5, 1, 2]
        ]
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':dyn_obs_list})

        ############### 4th Graph #############################
        boundary_coordinates = [(3.6, 57.8), (3.6, 3.0), (58.3, 3.0), (58.1, 58.3)]
        obstacle_list = [
            [(21.1, 53.1), (21.4, 15.1), (9.3, 15.1), (9.1, 53.1)],
            [(35.7, 52.2), (48.2, 52.3), (48.7, 13.6), (36.1, 13.8)], 
            [(17.0, 50.5),(30.7, 50.3), (30.6, 45.0), (17.5, 45.1)],
            [(26.4, 39.4), (40.4, 39.3), (40.5, 35.8), (26.3, 36.0)],
            [(19.3, 31.7), (30.3, 31.6), (30.1, 27.7), (18.9, 27.7)],
            [(26.9, 22.7), (41.4, 22.6), (41.1, 17.5), (27.4, 17.6)] ]
        start = (30,5, math.radians(90))
        end = (30,55, math.radians(90))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 5th Graph #############################
        boundary_coordinates = [(54.0, 57.8), (7.8, 57.5), (7.5, 17.9), (53.0, 17.0)]
        obstacle_list = [
            [(14.0, 57.6), (42.1, 57.6), (42.2, 52.0), (13.4, 52.0)], 
            [(7.7, 49.1), (32.2, 49.0), (32.1, 45.3), (7.7, 45.8)], 
            [(34.2, 53.0), (41.2, 53.1), (40.9, 31.7), (34.4, 31.9)], 
            [(35.7, 41.7), (35.7, 36.8), (11.7, 39.8), (12.1, 44.0), (31.3, 43.3)], 
            [(5.8, 37.6), (24.1, 35.0), (23.6, 29.8), (5.0, 31.8)], 
            [(27.1, 39.7), (32.7, 39.0), (32.8, 24.7), (16.2, 20.9), (14.5, 25.9), (25.3, 26.7), (27.9, 31.4), (26.1, 39.2)] ]
        start = (10.3, 55.8, math.radians(270))
        end = (38.1, 25.0, math.radians(300))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 6th Graph ############################# 
        boundary_coordinates = [(0.37, 0.32), (5.79, 0.31), (5.79, 5.18), (0.14, 5.26)]
        obstacle_list = [[(2.04, 0.28), (2.0, 3.8), (2.8, 3.81), (2.78, 0.29)]]
        start = (1.01, 0.98, math.radians(90))
        end = (3.82, 1.05, math.radians(270))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 7th Graph ############################# 
        # NOTE: Not always working.
        boundary_coordinates = [(1.55, 1.15), (29.0, 1.1), (29.0, 28.75), (0.85, 28.9), (0.85, 1.15)]
        obstacle_list = [[(5.6, 3.3), (5.75, 20.15), (18.35, 20.05), (18.35, 19.7), (7.25, 19.7), (7.05, 3.2)], [(13.85, 23.4), (21.25, 23.35), (21.1, 16.4), (6.9, 16.35), (6.7, 12.9), (23.45, 13.25), (23.4, 25.05), (13.0, 25.35)]]
        start = (2.95, 13.5, math.radians(90))
        end = (9.6, 18.1, math.radians(180))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 8th Graph #############################
        boundary_coordinates = [(2.0, 1.08), (22.8, 1.12), (22.84, 19.16), (1.8, 19.24)]
        obstacle_list = [[(9.64, 5.28), (9.56, 10.72), (8.68, 11.88), (9.48, 12.2), (10.52, 10.96), (11.6, 12.12), (12.6, 11.36), (11.28, 10.4), (11.6, 0.56), (9.68, 0.68)]]
        start = (7.16, 8.16, math.radians(90))
        end = (12.72, 9.32, math.radians(265))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 9th Graph #############################
        #NOTE: Not always working.
        boundary_coordinates = [(0.96, 1.88), (22.88, 1.72), (22.92, 20.8), (0.64, 20.92)]
        obstacle_list = [[(9.12, 1.48), (8.8, 9.56), (9.76, 12.72), (10.8, 9.56), (11.08, 1.48)]]
        start = (7.44, 6.16, math.radians(90))
        end = (12.44, 6.4, math.radians(265))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 10th Graph ############################# 
        boundary_coordinates = [(2.36, 1.6), (22.6, 1.84), (22.16, 21.04), (1.52, 20.88)]
        obstacle_list = [[(9.92, 1.24), (9.64, 8.52), (12.6, 10.44), (15.6, 8.76), (15.76, 1.08)]]
        start = (7.08, 5.88, math.radians(90))
        end = (17.8, 6.56, math.radians(265))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 11th Graph #############################
        boundary_coordinates = [(1.5, 1.0), (1.7, 58.6), (59.0, 58.4), (58.6, 1.3)]
        obstacle_list = [
            [(27.0, 6.0), (27.0, 33.0), (4.0, 33.0), (4.0, 6.0)], 
            [(65.0, 6.0), (28.1, 6.0), (28.1, 33.0), (65.0, 33.0)], 
            [(4.4, 34.1), (44.0, 34.1), (44.0, 39.3), (55.3, 39.6), (55.3, 42.8), (44.0, 42.3), (44.1, 49.1), (54.9, 49.2), (54.9, 53.0), (4.7, 53.0)], 
            [(47.7, 36.2), (47.7, 34.6), (57.8, 34.5), (57.8, 36.3)]]
        start = (27.8, 2.7, math.radians(90))
        end = (50.3, 45.9, math.radians(0))
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':[]})

        ############### 12th Graph #############################
        boundary_coordinates = [(11.9, 3.6), (11.9, 50.6), (47.3, 50.6), (47.3, 3.6)]
        obstacle_list = [[(11.9, 11.8), (22.2, 11.8), (22.2, 15.9), (11.9, 15.9)],
            [(11.9, 20.4), (22.2, 20.4), (22.2, 25.0), (11.9, 25.0)],
            [(28.0, 25.5), (28.0, 20.5), (32.4, 20.5), (32.4, 15.7), (28.0, 15.7), (28.0, 3.6), (37.8, 3.6), (37.8, 25.5)], # low
            [(15.9, 29), (37.7, 29), (37.7, 44.5), (25.3, 44.5), (25.3, 40.7), (35.0, 40.7), (35.0, 31.7), (15.9, 31.7)], # up
            [(29.8, 28.7), (29.8, 25.8), (34.5, 25.8), (34.5, 28.7)] ]
        start = (18.9, 7.0, math.radians(45))
        end = (44.7, 6.8, math.radians(270))
        # Starting point, ending point, freq, x radius, y radius, angle
        dyn_obs_list = [
            [[18.5, 18.2],[28.1, 18.2], 0.06, 0.5, 1.0, math.pi/2],
            [[16.775, 34.0], [22.5, 42.2], 0.07, 0.3, 0.7, math.pi/2+0.961299],
            [[44.3, 9.2], [40.5, 31.8], 0.0745, 0.6, 0.6, 0]
        ]
        self.graphs.append({'boundary_coordinates':boundary_coordinates, 'obstacle_list':obstacle_list, 'start':start, 'end':end, 'dyn_obs_list':dyn_obs_list})

    def get_graph(self, index):
        if not (0 <= index <= len(self.graphs)-1):
            raise ValueError(f'Graph index not found (should be in the range {0} - {len(self.graphs)-1})')
        return self.graphs[index]

if __name__ == '__main__':
    graph = Graph(index=11) # right now 0~11
    graph.plot_map()
