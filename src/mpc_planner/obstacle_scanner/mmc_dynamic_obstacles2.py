import os, sys

from typing import Tuple, List

import numpy as np
import pandas as pd

'''
File info:
    Name    - [mmc_dynamic_obstacles]
    Author  - [Ze]
    Date    - [May. 01, 2022] -> [??. ??, 2022]
    Exe     - [Yes]
File description:
    Load the dynamic obstacles and their future predictions from json files.
File content:
    ObstacleScanner <class> - Offering information and MPC-format list of dynamic obstacles.
Comments:
    This code is a disaster but works, so...
'''

class ObstacleScanner():
    '''
    Description:
        Generate/read the information of dynamic obstacles, and form it into the correct format for MPC.
    Attributes:
        num_obstacles <int> - The (maximal) number of (active) dynamic obstacles.
        num_modes     <int> - The (maximal) number of modes that each dynamic obstacle has.
    Functions
        get_obstacle_info      <get> - Get the position, orientation, and shape information of a selected (idx) obstacle.
        get_full_obstacle_list <get> - Form the list for MPC problem.
    Comments:
        Other attributes and functions are just for testing and specific in thie file.
    '''
    def __init__(self):
        self.ts = 0.2
        self.vehicle_width  = 0.5
        self.vehicle_margin = 0.25

        self.num_obstacles = 10    # fixed here
        self.num_modes     = 3     # fixed here
        self.pred_time_offset = 20 # fixed here

        self.obj_list = []

        self.create_pedestrian1(start_timestep=-7,  index=1, speed=1) # index: 1-crossing;   2-avoiding
        self.create_pedestrian2(start_timestep=-12, index=1, speed=1) # index: 1,3-crossing; 2-avoiding
        self.create_pedestrian3(start_timestep=20,   index=1, speed=1) # index: 1-crossing;   2,3-avoiding
        self.create_vehicle4(start_timestep=10,   index=2, speed=1) # index: 2,3-crossing; 1-avoiding
        self.create_vehicle5(start_timestep=-5,   index=2, speed=1) # index: 2,3-crossing; 1-avoiding

        self.obj_df = pd.DataFrame(self.obj_list)

    def get_obstacle_info(self, idx, current_time, key): # the start time of the first prediction
        current_obs_df = self.get_current_obs_df(current_time)
        if current_obs_df is None:
            obs_dict = {'position':(0,0), 'pos':(0,0), 'radius':(0,0), 'axis':(0,0), 'heading':0, 'angle':0}
        else:
            this_df = current_obs_df.loc[current_obs_df['id']==idx]
            rx = 0.2 # + self.vehicle_width/2 + self.vehicle_margin
            ry = 0.2 # + self.vehicle_width/2 + self.vehicle_margin
            obs_dict = {'position':this_df['p'], 'pos':this_df['p'], 'radius':(rx,ry), 'axis':(rx,ry), 'heading':0, 'angle':0}
        if key==-1:
            return obs_dict
        return obs_dict[key]

    def get_full_obstacle_list(self, current_time, horizon):
        current_obs_df = self.get_current_obs_df(current_time) # pred
        if current_obs_df is None:
            current_obs_df = self.get_current_obs_df(current_time) # pred
            return []

        obstacles_list = []
        for id in list(current_obs_df['id']):
            this_df = current_obs_df.loc[current_obs_df['id']==id]
            for n in range(self.num_modes):
                obs_list = []
                for T in range(horizon):
                    obs_pred = this_df[f'pred_T{T+1}'].item()
                    alpha, x, y, rx ,ry = obs_pred[n]
                    rx += self.vehicle_width/2 + self.vehicle_margin
                    ry += self.vehicle_width/2 + self.vehicle_margin
                    obs_T = (x, y, rx, ry, 0, alpha) # angle=0
                    obs_list.append(obs_T)
                obstacles_list.append(obs_list)
        return obstacles_list # x, y, rx, ry, theta for t in horizon

    def get_current_obs_df(self, current_time):
        time_step = int(current_time/self.ts)
        if time_step in list(self.obj_df['t']):
            current_obs_df = self.obj_df.loc[self.obj_df['t']==time_step]
            return current_obs_df # pred
        return None

    def moving_agent(self, start:np.ndarray, goal:np.ndarray, speed:float) -> Tuple[list, list]:
        speed *= self.ts
        distance = np.linalg.norm((goal-start))
        total_time_steps = int(distance / speed)
        x_coord = np.linspace(start=start[0], stop=goal[0], num=total_time_steps)
        y_coord = np.linspace(start=start[1], stop=goal[1], num=total_time_steps)
        return x_coord.tolist(), y_coord.tolist()

    def moving_agent_waypoints(self, waypoints:np.ndarray, speed:float) -> Tuple[list, list]:
        '''
        Args:
            waypoints: n*2, every row is a pair of waypoint coordinates
        '''
        x_coord = []
        y_coord = []
        for i in range(waypoints.shape[0]-1):
            x_part, y_part = self.moving_agent(start=waypoints[i,:], goal=waypoints[i+1,:], speed=speed)
            x_coord += x_part
            y_coord += y_part
        return x_coord, y_coord

    # Just for MMC
    def create_pedestrian1(self, start_timestep:int=0, speed:float=1.2, index:int=1):
        '''
        Description:
            Generate the dynamic obstacles and predictions with hard-coding.
        Arguments:
            start_timestep <int> - the start time step when the object becomes active
            index          <int> - the index of mode
        Return:
            obj_list <list of dicts> - Each dictionary contains all info of the target at a time instant.
        Comments:
            The format of the dictionary is
                {'t':t1, 'id':id, 'p':[x,y], 'pred_T1':[[a1,x1,y1,sx1,sy1], ..., [am,xm,ym,sxm,sym]], 'pred_T2':..., ...}
                where 'm' (m<m_max) is the number of components/futures and 'a' is the weight.
        '''
        assert (index in [1,2]), (f'Mode {index} not defined.')
        T = self.pred_time_offset
        sigma_x = 0.2
        sigma_y = 0.2
        a1 = 0.5 # do crossing
        a2 = 0.5 # no crossing

        x_, y_   = self.moving_agent(np.array([12,  3.5]), np.array([8.5, 3.5]), speed)
        x_1, y_1 = self.moving_agent(np.array([8.5, 3.5]), np.array([0,   3.5]), speed)
        x_2, y_2 = self.moving_agent(np.array([8.5, 3.5]), np.array([8.5, 12]),  speed)
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2
        y_m2 = y_ + y_2

        x_m_list = [x_m1, x_m2]
        y_m_list = [y_m1, y_m2]
        for k in range(0, len(x_m_list[index-1])-1):
            this_dict = {'t':k+start_timestep, 'id':1, 'p':[x_m_list[index-1][k], y_m_list[index-1][k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], x_m_list, y_m_list):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, x_m_list[index-1][k+i+1], y_m_list[index-1][k+i+1], sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    except:
                        this_dict[key] = [[1, x_m_list[index-1][-1],    y_m_list[index-1][-1],    sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

    def create_pedestrian2(self, start_timestep:int=0, speed:float=1.2, index:int=1):
        assert (index in [1,2,3]), (f'Mode {index} not defined.')
        T = self.pred_time_offset
        sigma_x = 0.2
        sigma_y = 0.2
        a1 = 0.5 # do crossing
        a2 = 0.5 # no crossing

        x_, y_   = self.moving_agent(np.array([8.5, 0]),   np.array([8.5, 3.6]), speed)
        x_1, y_1 = self.moving_agent(np.array([8.5, 3.6]), np.array([0,   3.6]), speed)
        x_2, y_2 = self.moving_agent(np.array([8.5, 3.6]), np.array([8.5, 8.5]), speed)
        x_2_1, y_2_1 = self.moving_agent(np.array([8.5, 8.5]), np.array([8.5, 12]), speed)
        x_2_2, y_2_2 = self.moving_agent(np.array([8.5, 8.5]), np.array([0, 8.5]), speed)
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2 + x_2_1
        y_m2 = y_ + y_2 + y_2_1
        x_m3 = x_ + x_2 + x_2_2
        y_m3 = y_ + y_2 + y_2_2

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]
        for k in range(0, len(x_m_list[index-1])-1):
            this_dict = {'t':k+start_timestep, 'id':2, 'p':[x_m_list[index-1][k], y_m_list[index-1][k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m1, x_m2], [y_m1, y_m2]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            elif (index!=1) and (k < (len(x_)+len(x_2))):
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m2, x_m3], [y_m2, y_m3]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, x_m_list[index-1][k+i+1], y_m_list[index-1][k+i+1], sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    except:
                        this_dict[key] = [[1, x_m_list[index-1][-1],    y_m_list[index-1][-1],    sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

    def create_pedestrian3(self, start_timestep:int=0, speed:float=1.2, index:int=1):
        assert (index in [1,2,3]), (f'Mode {index} not defined.')
        T = self.pred_time_offset
        sigma_x = 0.2
        sigma_y = 0.2
        a1 = 0.4 # do crossing
        a2 = 0.3 # no crossing
        a3 = 0.3 # no crossing

        x_, y_   = self.moving_agent(np.array([12,  8.5]), np.array([8.3, 8.5]), speed)
        x_1, y_1 = self.moving_agent(np.array([8.3, 8.5]), np.array([0, 8.5]),   speed)
        x_2, y_2 = self.moving_agent(np.array([8.3, 8.5]), np.array([8.3, 12]),  speed)
        x_3, y_3 = self.moving_agent(np.array([8.3, 8.5]), np.array([8.3, 0]),   speed)
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2
        y_m2 = y_ + y_2
        x_m3 = x_ + x_3
        y_m3 = y_ + y_3

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]
        for k in range(0, len(x_m_list[index-1])-1):
            this_dict = {'t':k+start_timestep, 'id':3, 'p':[x_m_list[index-1][k], y_m_list[index-1][k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2, a3], x_m_list, y_m_list):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, x_m_list[index-1][k+i+1], y_m_list[index-1][k+i+1], sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    except:
                        this_dict[key] = [[1, x_m_list[index-1][-1],    y_m_list[index-1][-1],    sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

    def create_vehicle4(self, start_timestep:int=0, speed:float=1.2, index:int=1):
        assert (index in [1,2,3]), (f'Mode {index} not defined.')
        T = self.pred_time_offset
        sigma_x = 0.4
        sigma_y = 0.4
        a1 = 0.5
        a2 = 0.5

        x_, y_   = self.moving_agent(np.array([12, 7]), np.array([9, 7]), speed)
        x_1, y_1 = self.moving_agent_waypoints(np.array([[9, 7], [7, 7], [7, 12]]), speed) # turn right
        x_2, y_2 = self.moving_agent(np.array([9, 7]), np.array([5, 7]), speed)
        x_2_1, y_2_1 = self.moving_agent(np.array([5, 7]), np.array([5, 0]), speed) # turn left
        x_2_2, y_2_2 = self.moving_agent(np.array([5, 7]), np.array([0, 7]), speed) # go straight
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2 + x_2_1
        y_m2 = y_ + y_2 + y_2_1
        x_m3 = x_ + x_2 + x_2_2
        y_m3 = y_ + y_2 + y_2_2

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]
        for k in range(0, len(x_m_list[index-1])-1):
            this_dict = {'t':k+start_timestep, 'id':4, 'p':[x_m_list[index-1][k], y_m_list[index-1][k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m1, x_m2], [y_m1, y_m2]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            elif (index!=1) and (k < (len(x_)+len(x_2))):
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m2, x_m3], [y_m2, y_m3]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, x_m_list[index-1][k+i+1], y_m_list[index-1][k+i+1], sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    except:
                        this_dict[key] = [[1, x_m_list[index-1][-1],    y_m_list[index-1][-1],    sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

    def create_vehicle5(self, start_timestep:int=0, speed:float=1.2, index:int=1):
        assert (index in [1,2,3]), (f'Mode {index} not defined.')
        T = self.pred_time_offset
        sigma_x = 0.4
        sigma_y = 0.4
        a1 = 0.5
        a2 = 0.5

        x_, y_   = self.moving_agent(np.array([0, 5]), np.array([3, 5]), speed)
        x_1, y_1 = self.moving_agent_waypoints(np.array([[3, 5], [5, 5], [5, 0]]), speed) # turn right
        x_2, y_2 = self.moving_agent(np.array([3, 5]), np.array([7, 5]), speed)
        x_2_1, y_2_1 = self.moving_agent(np.array([7, 5]), np.array([7, 12]), speed) # turn left
        x_2_2, y_2_2 = self.moving_agent(np.array([7, 5]), np.array([12, 5]), speed) # go straight
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2 + x_2_1
        y_m2 = y_ + y_2 + y_2_1
        x_m3 = x_ + x_2 + x_2_2
        y_m3 = y_ + y_2 + y_2_2

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]
        for k in range(0, len(x_m_list[index-1])-1):
            this_dict = {'t':k+start_timestep, 'id':5, 'p':[x_m_list[index-1][k], y_m_list[index-1][k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m1, x_m2], [y_m1, y_m2]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            elif (index!=1) and (k < (len(x_)+len(x_2))):
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m2, x_m3], [y_m2, y_m3]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T])
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, x_m_list[index-1][k+i+1], y_m_list[index-1][k+i+1], sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    except:
                        this_dict[key] = [[1, x_m_list[index-1][-1],    y_m_list[index-1][-1],    sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
                    this_dict[key] += [[0, 0, 0, 1, 1]] * (self.num_modes-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list



if __name__ == '__main__':

    reader = ObstacleScanner()

    info = reader.get_obstacle_info(0, current_time=0, key=-1)
    print(info)
    reader.get_full_obstacle_list(current_time=0, horizon=20, ts=0.2)

