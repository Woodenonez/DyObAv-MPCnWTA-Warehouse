import math
import numpy as np

from typing import List, Tuple


def normalize_theta(theta):
    """Normalize an angle in radians to [-pi, pi]"""
    return ((theta + math.pi) % (2 * math.pi)) - math.pi

def average_angle(theta_1: float, theta_2: float):
    """Compute the average of two angles in radians"""
    x = math.cos(theta_1) + math.cos(theta_2)
    y = math.sin(theta_1) + math.sin(theta_2)
    return math.atan2(y, x)

def estimateDeltaT(start_state: np.ndarray, end_state: np.ndarray, max_vel_x: float, max_vel_theta: float):
    """Estimate the time required to move from start_state to end_state with constant velocity"""
    dt_constant_motion = 0.1
    if max_vel_x > 0:
        trans_dist = np.linalg.norm(end_state[:2] - start_state[:2])
        dt_constant_motion = trans_dist / max_vel_x
    if max_vel_theta > 0:
        rot_dist = abs(normalize_theta(end_state[2] - start_state[2]))
        dt_constant_motion = max(dt_constant_motion, rot_dist / max_vel_theta)
    return dt_constant_motion

class TimedElasticBand:
    """The timed elastic band is a path representation that is used in the elastic band approach.
    
    Attributes:
        pose_vec: A list of poses (x, y, theta, fixed:bool) that represent the path.
        timediff_vec: A list of time differences (dt, fixed:bool) that represent the time differences between the poses.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        """Reset the path."""
        self.pose_vec = []
        self.timediff_vec = []
        
    def isInit(self) -> bool:
        return (len(self.pose_vec)>0) and (len(self.timediff_vec)>0)

    def add_pose(self, x: float, y: float, theta: float, fixed: bool):
        """Add a pose (x, y, theta, fixed:bool) to the path."""
        self.pose_vec.append([x, y, theta, fixed])

    def add_timediff(self, dt: float, fixed: bool):
        """Add a time difference dt to the path."""
        self.timediff_vec.append([dt, fixed])

    def add_pose_and_timediff(self, x: float, y: float, angle: float, dt: float, fixed:bool=False):
        if len(self.pose_vec) != len(self.timediff_vec):
            self.add_pose(x, y, angle, fixed)
            self.add_timediff(dt, fixed)
        else:
            raise Exception("There should be one more pose than timediff")
        
    def delete_pose(self, start_index: int, num_poses:int=1):
        """Delete a pose at index."""
        if len(self.pose_vec) < start_index + num_poses:
            raise Exception("Index out of range")
        for i in range(num_poses):
            self.pose_vec.pop(start_index+i)

    def delete_timediff(self, start_index: int, num_timediffs:int=1):
        """Delete a timediff at index."""
        if len(self.timediff_vec) < start_index + num_timediffs:
            raise Exception("Index out of range")
        for i in range(num_timediffs):
            self.timediff_vec.pop(start_index+i)

    def insert_pose(self, start_index: int, x: float, y: float, theta: float, fixed: bool):
        """Insert a pose (x, y, theta, fixed:bool) at index."""
        self.pose_vec.insert(start_index, [x, y, theta, fixed])

    def insert_timediff(self, start_index: int, dt: float, fixed: bool):
        """Insert a timediff dt at index."""
        self.timediff_vec.insert(start_index, [dt, fixed])

    def set_pose_status(self, index: int, fixed: bool):
        """Set the status of the pose at index to fixed."""
        self.pose_vec[index][3] = fixed

    def set_timediff_status(self, index: int, fixed: bool):
        """Set the status of the timediff at index to fixed."""
        self.timediff_vec[index][1] = fixed


    def auto_resize(self, dt_ref: float, dt_hysteresis: float, min_samples: int, max_samples: int, fast_mode: bool):
        assert len(self.timediff_vec) == 0 or len(self.timediff_vec)+1 == len(self.pose_vec)
        modified = True

        for _ in range(100):  # Not get stuck in some oscillation, max 100 repetitions.
            if not modified:
                break

            modified = False

            i = 0
            while i < len(self.timediff_vec):
                if self.timediff_vec[i][0] > dt_ref + dt_hysteresis and len(self.timediff_vec) < max_samples:
                    if self.timediff_vec(i) > 2 * dt_ref:
                        newtime = 0.5 * self.timediff_vec[i][0]

                        self.timediff_vec[i][0] = newtime
                        self.insert_pose(i+1, 
                                         x=(self.pose_vec[i][0] + self.pose_vec[i+1][0]) / 2,
                                         y=(self.pose_vec[i][1] + self.pose_vec[i+1][1]) / 2,
                                         theta=average_angle(self.pose_vec[i][2], self.pose_vec[i+1][2]))
                        self.insert_timediff(i+1, newtime)

                        i -= 1  # check the updated pose diff again
                        modified = True
                    else:
                        if i < len(self.timediff_vec)-1:
                            self.timediff_vec[i+1][0] += self.timediff_vec[i][0] - dt_ref
                        self.timediff_vec[i][0] = dt_ref
                elif self.timediff_vec[i][0] < dt_ref-dt_hysteresis and len(self.timediff_vec) > min_samples:
                    if i < len(self.timediff_vec) - 1:
                        self.timediff_vec[i+1][0] += self.timediff_vec[i][0]
                        self.delete_timediff(i)
                        self.delete_pose(i+1)
                        i -= 1  # check the updated pose diff again
                    else:
                        self.timediff_vec[i-1][0] += self.timediff_vec[i][0]
                        self.delete_timediff(i)
                        self.delete_pose(i)

                    modified = True
                i += 1

            if fast_mode:
                break

    def sum_timediffs(self):
        """Return the sum of all timediffs."""
        return sum([timediff[0] for timediff in self.timediff_vec])
    
    def sum_timediffs_upto(self, index: int):
        """Return the sum of all timediffs up to index."""
        assert index < len(self.timediff_vec)
        return sum([timediff[0] for timediff in self.timediff_vec[:index]])
    
    def sum_distance(self):
        """Return the sum distance from the first pose to the last pose."""
        dist = 0
        for i in range(1, len(self.pose_vec)):
            dist += np.linalg.norm(np.array(self.pose_vec[i][:2]) - np.array(self.pose_vec[i-1][:2]))
        return dist
    
    def init_trajectory_to_goal(self, start: tuple, goal: tuple, diststep: float, max_vel_x: float, min_samples: int, guess_backwards_motion: bool):
        """Initialize a trajectory from start to goal with a fixed distance between poses.
        
        Arguments:
            start: (x, y, theta).
            goal: (x, y, theta)."""
        if not self.isInit():
            self.add_pose(*start, fixed=True)

            timestep = 0.1

            if diststep != 0:
                point_to_goal = np.array(goal[:2]) - np.array(start[:2])
                dir_to_goal = math.atan2(point_to_goal[1], point_to_goal[0])
                dx = diststep * math.cos(dir_to_goal)
                dy = diststep * math.sin(dir_to_goal)
                orient_init = dir_to_goal

                if guess_backwards_motion and np.dot(point_to_goal, np.array([np.cos(start[2]), np.sin(start[2])])) < 0:
                    orient_init = normalize_theta(orient_init + math.pi)

                dist_to_goal = np.linalg.norm(point_to_goal)
                no_steps_d = dist_to_goal / abs(diststep)
                no_steps = math.floor(no_steps_d)

                if max_vel_x > 0:
                    timestep = diststep / max_vel_x

                for i in range(1, no_steps + 1):
                    if i == no_steps and no_steps_d == float(no_steps):
                        break
                    self.add_pose_and_timediff(start[i][0]+i*dx, start[i][1]+i*dy, orient_init, timestep)

            if len(self.pose_vec) < min_samples - 1:
                while len(self.pose_vec) < min_samples - 1:                        
                    intermediate_pose = [(self.pose_vec[-1][0] + goal[0]) / 2,
                                         (self.pose_vec[-1][1] + goal[1]) / 2,
                                         average_angle(self.pose_vec[-1][2], goal[2])]
                    if max_vel_x > 0:
                        timestep = np.linalg.norm(np.array(intermediate_pose[:2]) - np.array(self.pose_vec[-1][:2])) / max_vel_x
                    self.add_pose_and_timediff(*intermediate_pose, timestep)

            if max_vel_x > 0:
                timestep = np.linalg.norm(np.array(goal[:2]) - np.array(self.pose_vec[-1][:2])) / max_vel_x
            self.add_pose_and_timediff(*goal, timestep, fixed=True)
        else:
            print("Cannot init TEB between given configuration and goal, because TEB vectors are not empty or TEB is already initialized (call this function before adding states yourself)!")
            print(f"Number of TEB configurations: {len(self.pose_vec)}, Number of TEB timediffs: {len(self.timediff_vec)}")
            return False

        return True

    def init_trajectory_to_goal_from_plan(self, plan: List[tuple], max_vel_x: float, max_vel_theta: float, estimate_orient: bool, min_samples: int, guess_backwards_motion: bool):
        """Initialize a trajectory from a plan.
        
        Arguments:
            plan: [(x, y, theta), ...] (list of tuples).
        """
        if not self.isInit():
            start = plan[0][:3]
            goal = plan[-1][:3]

            self.add_pose(*start, fixed=True)

            backwards = False
            if guess_backwards_motion and np.dot(np.array(goal[:2]) - np.array(start[:2]), np.array([np.cos(start[2]), np.sin(start[2])])) < 0:  # check if the goal is behind the start pose (w.r.t. start orientation)
                backwards = True
            # TODO: dt ~ max_vel_x_backwards for backwards motions

            for i in range(1, len(plan) - 1):
                if estimate_orient:
                    # get yaw from the orientation of the distance vector between pose_{i+1} and pose_{i}
                    dx = plan[i+1][0] - plan[i][0]
                    dy = plan[i+1][1] - plan[i][1]
                    yaw = np.arctan2(dy, dx)
                    if backwards:
                        yaw = normalize_theta(yaw + np.pi)
                else:
                    yaw = plan[i][3]
                
                intermediate_pose = [plan[i][0], plan[i][1], yaw]
                dt = estimateDeltaT(np.array(self.pose_vec[-1]), np.array(intermediate_pose), max_vel_x, max_vel_theta)
                self.add_pose_and_timediff(*intermediate_pose, dt)

            # if number of samples is not larger than min_samples, insert manually
            if len(self.pose_vec) < min_samples - 1:
                print("initTEBtoGoal(): number of generated samples is less than specified by min_samples. Forcing the insertion of more samples...")
                while len(self.pose_vec) < min_samples - 1:  # subtract goal point that will be added later
                    # simple strategy: interpolate between the current pose and the goal
                    intermediate_pose = [(self.pose_vec[-1][0] + goal[0]) / 2,
                                         (self.pose_vec[-1][1] + goal[1]) / 2,
                                         average_angle(self.pose_vec[-1][2], goal[2])]
                    dt = estimateDeltaT(np.array(self.pose_vec[-1]), np.array(intermediate_pose), max_vel_x, max_vel_theta)
                    self.add_pose_and_timediff(*intermediate_pose, dt)  # let the optimizer correct the timestep (TODO: better initialization)

            # Now add final state with given orientation
            dt = estimateDeltaT(np.array(self.pose_vec[-1]), np.array(goal), max_vel_x, max_vel_theta)
            self.add_pose_and_timediff(*goal, dt)
            self.set_pose_status(len(self.pose_vec) - 1, True)  # GoalConf is a fixed constraint during optimization
        else:  # size!=0
            print(f"Cannot init TEB between given configuration and goal, because TEB vectors are not empty or TEB is already initialized (call this function before adding states yourself)!")
            print(f"Number of TEB configurations: {len(self.pose_vec)}, Number of TEB timediffs: {len(self.timediff_vec)}")
            return False

        return True

    def find_closest_trajectory_pose(self, ref_point: np.ndarray, begin_idx: int):
        n = len(self.pose_vec)
        if begin_idx < 0 or begin_idx >= n:
            return -1, None

        min_dist_sq = np.inf
        min_idx = -1

        for i in range(begin_idx, n):
            dist_sq = np.linalg.norm(ref_point - np.array(self.pose_vec[i][:2]))**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                min_idx = i

        distance = np.sqrt(min_dist_sq)

        return min_idx, distance
    
    def update_and_prune_teb(self, new_start:tuple=None, new_goal:tuple=None, min_samples:int=0):
        """Update the trajectory with new start and goal poses and prune the trajectory.
        
        Arguments:
            new_start: (x, y, theta) tuple or None.
            new_goal: (x, y, theta) tuple or None.
        """

        if new_start and len(self.pose_vec) > 0:
            # find nearest state (using l2-norm) in order to prune the trajectory
            # (remove already passed states)
            dist_cache = np.linalg.norm(np.array(new_start[:2]) - np.array(self.pose_vec[0][:2]))
            
            lookahead = min(len(self.pose_vec) - min_samples, 10)  # satisfy min_samples, otherwise max 10 samples

            nearest_idx = 0
            for i in range(1, lookahead+1):
                dist = np.linalg.norm(np.array(new_start[:2]) - np.array(self.pose_vec[i][:2]))
                if dist < dist_cache:
                    dist_cache = dist
                    nearest_idx = i
                else:
                    break

            # prune trajectory at the beginning (and extrapolate sequences at the end if the horizon is fixed)
            if nearest_idx > 0:
                # nearest_idx is equal to the number of samples to be removed (since it counts from 0 ;-) )
                # WARNING delete starting at pose 1, and overwrite the original pose(0) with new_start, since Pose(0) is fixed during optimization!
                self.delete_pose(1, nearest_idx)  # delete first states such that the closest state is the new first one
                self.delete_timediff(1, nearest_idx)  # delete corresponding time differences
            
            # update start
            self.pose_vec[0] = new_start

        if new_goal and len(self.pose_vec) > 0:
            self.pose_vec[:3] = new_goal

    def is_trajectory_inside_region(self, radius: float, max_dist_behind_robot: float, skip_poses: int):
        if len(self.pose_vec) <= 0:
            return True

        radius_sq = radius * radius
        max_dist_behind_robot_sq = max_dist_behind_robot * max_dist_behind_robot
        robot_orient = np.array([np.cos(self.pose_vec[0][2]), np.sin(self.pose_vec[0][2])])

        for i in range(1, len(self.pose_vec), skip_poses + 1):
            dist_vec = np.array(self.pose_vec[i][:2]) - np.array(self.pose_vec[0][:2])
            dist_sq = np.dot(dist_vec, dist_vec)  # squared norm

            if dist_sq > radius_sq:
                print("outside robot")
                return False

            # check behind the robot with a different distance, if specified (or >=0)
            if max_dist_behind_robot >= 0 and np.dot(dist_vec, robot_orient) < 0 and dist_sq > max_dist_behind_robot_sq:
                print("outside robot behind")
                return False

        return True



