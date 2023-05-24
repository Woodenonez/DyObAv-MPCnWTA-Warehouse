import math
import statistics
import numpy as np

from shapely.geometry import Point, Polygon

from typing import List, Tuple

HUMAN_SIZE = 0.2

def check_collision(state: np.ndarray, 
                    static_obstacles: List[List[tuple]], 
                    dynamic_obstacle: List[tuple]) -> bool:
    pos = Point(state[0], state[1])
    ### Check static obstacles
    for obstacle in static_obstacles:
        if Polygon(obstacle).contains(pos):
            return True
    ### Check dynamic obstacles
    for obstacle in dynamic_obstacle:
        if pos.distance(Point(obstacle[0], obstacle[1])) <= HUMAN_SIZE:
            return True
    return False

def calc_action_smoothness(action_list: List[np.ndarray]):
    speeds = np.array(action_list)[:, 0]
    angular_speeds = np.array(action_list)[:, 1]
    return [statistics.mean(np.abs(np.diff(speeds, n=2))), statistics.mean(np.abs(np.diff(angular_speeds, n=2)))]

def calc_minimal_obstacle_distance(trajectory: List[tuple], obstacles: List[List[tuple]]):
    dist_list = []
    for pos in trajectory:
        dist_list.append(min([Polygon(obs).distance(Point(pos)) for obs in obstacles]))
    return min(dist_list)

def calc_minimal_dynamic_obstacle_distance(state: np.ndarray, obstacles: List[tuple]):
    dist_list = [np.linalg.norm(state[:2]-np.array(obstacle)) for obstacle in obstacles]
    return min(dist_list)

def calc_deviation_distance(ref_traj: List[tuple], actual_traj: List[tuple]):
    deviation_dists = []
    for pos in actual_traj:
        deviation_dists.append(min([math.hypot(ref_pos[0]-pos[0], ref_pos[1]-pos[1]) for ref_pos in ref_traj]))
    return [statistics.mean(deviation_dists), max(deviation_dists)]



