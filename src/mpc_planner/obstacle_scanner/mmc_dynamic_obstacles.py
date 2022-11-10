import os, sys
import numpy as np

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
        num_obstacles <int> - The number of (active) dynamic obstacles.
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

        self.num_obstacles = 1 # fixed here
        self.start_time = -5 * self.ts # -7

        self.create_pedestrian(index=2)

    def get_obstacle_info(self, idx, current_time, key): # the start time of the first prediction
        obs_dict = self.get_obs_dict(current_time)
        if obs_dict is None:
            obs_dict = {'position':(0,0), 'pos':(0,0), 'radius':(0,0), 'axis':(0,0), 'heading':0, 'angle':0}
        else:
            pos = (obs_dict['info'][1],obs_dict['info'][2])
            rx = 0.2 # + self.config.vehicle_width/2 + self.config.vehicle_margin
            ry = 0.2 # + self.config.vehicle_width/2 + self.config.vehicle_margin
            obs_dict = {'position':pos, 'pos':pos, 'radius':(rx,ry), 'axis':(rx,ry), 'heading':0, 'angle':0}
        if key==-1:
            return obs_dict
        return obs_dict[key]

    def get_full_obstacle_list(self, current_time, horizon, ts=0):
        num_compo = 2
        obs_dict = self.get_obs_dict(current_time) # pred
        if obs_dict is None:
            return []

        obstacles_list = []
        for n in range(num_compo):
            obs_list = []
            for T in range(horizon):
                try:
                    obs_pred = obs_dict[list(obs_dict)[T+1]][n]
                except:
                    obs_pred = obs_dict[list(obs_dict)[T+1]][0]
                alpha, x, y, rx ,ry = obs_pred
                rx += self.vehicle_width/2 + self.vehicle_margin
                ry += self.vehicle_width/2 + self.vehicle_margin
                obs_T = (x, y, rx, ry, 0, alpha) # angle=0
                obs_list.append(obs_T)
            obstacles_list.append(obs_list)
        return obstacles_list # x, y, rx, ry, theta, (alpha) for t in horizon

    # Just for MMC
    def create_pedestrian(self, index:int=1):
        '''
        Description:
            Generate the dynamic obstacles and predictions with hard-coding.
        Arguments:
            index <int> - the index of mode, 1 (no crossing) or 2 (do crossing).
        Return:
            obj_list <list of dicts> - Each dictionary contains all info of the target at a time instant.
        Comments:
            The format of the dictionary is
                {'info':[t1,x,y], 'pred_T1':[[a1,x1,y1,sx1,sy1], ..., [am,xm,ym,sxm,sym]], 'pred_T2':..., ...}
                where 'm' is the number of components/futures and 'a' is the weight.
        '''
        assert (index in [1,2]), (f'Mode {index} not defined.')
        speed = 1.2
        speed *= self.ts
        T = 20 # prediction time offset
        sigma_x = 0.2
        sigma_y = 0.2
        a1 = 0.5 # no crossing
        a2 = 0.5 # do crossing
        self.obj_list = []

        x_before_crossing = np.arange(start=0, stop=10, step=speed).tolist()
        y_before_crossing = [0.8] * len(x_before_crossing)
        x_m1 = np.arange(start=10+speed, stop=16, step=speed).tolist() # no crossing
        y_m1 = y_before_crossing + [0.8] * len(x_m1) # no crossing
        x_m1 = x_before_crossing + x_m1
        y_m2 = np.arange(start=0.8+speed, stop=9, step=speed).tolist() # do crossing
        x_m2 = x_before_crossing + [10] * len(y_m2) # do crossing
        y_m2 = y_before_crossing + y_m2
        len_diff = abs(len(x_m1)-len(x_m2))
        if len(x_m1)>len(x_m2):
            x_m2 += [x_m2[-1]]*len_diff
            y_m2 += [y_m2[-1]]*len_diff
        else:
            x_m1 += [x_m1[-1]]*len_diff
            y_m1 += [y_m1[-1]]*len_diff
        x_m1 += [x_m1[-1]]*100
        y_m1 += [y_m1[-1]]*100
        x_m2 += [x_m2[-1]]*100
        y_m2 += [y_m2[-1]]*100

        this_x = [x_m1, x_m2][index-1]
        this_y = [y_m1, y_m2][index-1]
        for k in range(0, len(this_x)-1-T):
            this_dict = {'info':[k, this_x[k], this_y[k]]}
            if k < len(x_before_crossing): # with two potential modes [k < len(x_before_crossing)]
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = [[a1, x_m1[k+i+1], y_m1[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T], 
                                      [a2, x_m2[k+i+1], y_m2[k+i+1], sigma_x*(2)*(i+1)/T, sigma_y*(2)*(i+1)/T]]
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = [[1, this_x[k+i+1], this_y[k+i+1], sigma_x*2*(i+1)/T, sigma_y*2*(i+1)/T]]
            self.obj_list.append(this_dict)
        return self.obj_list

    def get_obs_dict(self, current_time):
        if current_time >= self.start_time:
            time_step = int((current_time-self.start_time)/self.ts)
            if time_step <= len(list(self.obj_list))-1:
                obs_dict = self.obj_list[time_step]
                return obs_dict # pred
        return None
        


