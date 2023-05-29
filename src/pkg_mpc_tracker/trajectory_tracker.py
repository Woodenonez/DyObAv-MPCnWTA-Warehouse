# System import
import os
import sys
import math
# External import
import numpy as np
# Custom import 
from configs import MpcConfiguration, CircularRobotSpecification
# Type hint
from typing import Callable, Tuple, Union, List, Dict, Any


class Solver(): # this is not found in the .so file (in ternimal: nm -D  navi_test.so)
    import opengen as og
    def run(self, p: list, initial_guess, initial_lagrange_multipliers, initial_penalty) -> og.opengen.tcp.solver_status.SolverStatus: pass


class TrajectoryTracker:
    """Generate a smooth trajectory tracking based on the reference path and obstacle information.
    
    Attributes:
        config: MPC configuration.
        robot_spec: Robot specification.

    Functions:
        run_step: Run one step of trajectory tracking.

    Comments:
        The solver needs to be built before running the trajectory tracking. \n
        To run the tracker: 
            1. Load motion model and init states; 
            2. Set reference path and trajectory (and states maybe);
            3. Run step.
    """
    def __init__(self, config: MpcConfiguration, robot_specification: CircularRobotSpecification, use_tcp:bool=False, verbose=False):
        self._prt_name = '[TrajTracker]'
        self.vb = verbose
        self.config = config
        self.robot_spec = robot_specification

        # Common used parameters from config
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor

        # Initialization
        self.idle = True
        self.set_work_mode(mode='safe')
        self.set_obstacle_weights(stc_weights=10, dyn_weights=10) # default obstacle weights

        self.__import_solver(use_tcp=use_tcp)

    def __import_solver(self, root_dir:str='', use_tcp:bool=False):
        self.use_tcp = use_tcp
        solver_path = os.path.join(root_dir, self.config.build_directory, self.config.optimizer_name)

        import opengen as og
        if not use_tcp:
            sys.path.append(solver_path)
            built_solver = __import__(self.config.optimizer_name) # it loads a .so (shared library object)
            self.solver:Solver = built_solver.solver() # Return a Solver object with run method, cannot find it though
        else: # use TCP manager to access solver
            self.mng:og.opengen.tcp.OptimizerTcpManager = og.tcp.OptimizerTcpManager(solver_path)
            self.mng.start()
            self.mng.ping() # ensure RUST solver is up and runnings

    def load_motion_model(self, motion_model: Callable) -> None:
        """The motion model should be `s'=f(s,a,ts)` (takes in a state and an action and returns the next state).
        """
        self.motion_model = motion_model

    def load_init_states(self, current_state: np.ndarray, goal_state: np.ndarray):
        """Load the initial state and goal state.

        Arguments:
            current_state: Current state of the robot.
            goal_state: Goal state of the robot.

        Attributes:
            state: Current state of the robot.
            final_goal: Goal state of the robot.
            past_states: List of past states of the robot.
            past_actions: List of past actions of the robot.
            cost_timelist: List of cost values of the robot.
            solver_time_timelist: List of solver time of the robot.

        Comments:
            This function resets the `idx_ref_traj/path` to 0 and `idle` to False.
        """
        if (not isinstance(current_state, np.ndarray)) or (not isinstance(goal_state, np.ndarray)):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}/{type(goal_state)}.')
        self.state = current_state
        self.final_goal = goal_state # used to check terminal condition

        self.past_states  = []
        self.past_actions: List[np.ndarray] = []
        self.cost_timelist = []
        self.solver_time_timelist = []

        self.idx_ref_traj = 0 # for reference trajectory following
        self.idx_ref_path = 0 # for reference path following
        self.idle = False

    def set_obstacle_weights(self, stc_weights: Union[list, int], dyn_weights: Union[list, int]):
        """
        Attributes:
            stc_weights [list]: penalty weights for static obstacles (only useful if soft constraints activated)
            dyn_weights [list]: penalty weights for dynamic obstacles (only useful if soft constraints activated)
        """
        if isinstance(stc_weights, list):
            self.stc_weights = stc_weights
        elif isinstance(stc_weights, (float,int)):
            self.stc_weights = [stc_weights]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(stc_weights)}.')
        if isinstance(dyn_weights, list):
            self.dyn_weights = dyn_weights
        elif isinstance(dyn_weights, (float,int)):
            self.dyn_weights = [dyn_weights]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(dyn_weights)}.')

    def set_work_mode(self, mode:str='safe'): # change base speed and tuning parameters
        """Set the basic work mode (base speed and weight parameters) of the MPC solver.
        Arguments:
            `mode`: "aligning" (start) or "safe" (20% speed) or "work" (80% speed) or "super" (full speed)
        Attributes:
            `base_speed`: The reference speed
            `tuning_params`: Penalty parameters for MPC
        """
        ### Base/reference speed
        if mode == 'aligning':
            self.base_speed = self.robot_spec.lin_vel_max*0.5
            self.tuning_params = [0.0] * self.config.nq
            self.tuning_params[2] = 100
        else:
            self.tuning_params = [self.config.qpos, self.config.qvel, self.config.qtheta, self.config.lin_vel_penalty, self.config.ang_vel_penalty,
                                  self.config.qpN, self.config.qthetaN, self.config.qrpd, self.config.lin_acc_penalty, self.config.ang_acc_penalty]
            if mode == 'safe':
                self.base_speed = self.robot_spec.lin_vel_max*0.2
            elif mode == 'work':
                self.base_speed = self.robot_spec.lin_vel_max*0.8
            elif mode == 'super':
                self.base_speed = self.robot_spec.lin_vel_max*1.0
            else:
                raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def set_current_state(self, current_state: np.ndarray):
        """To synchronize the current state of the robot with the MPC solver."""
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

    def set_ref_trajectory(self, ref_path: List[tuple], ref_traj:List[tuple]=None):
        """Set the global reference trajectory.

        Arguments:
            ref_path: Global reference path, must be given.
            ref_traj: Global reference trajectory, if not given, generated by the simple `get_ref_traj` method.

        Attributes:
            idx_ref_path: Index of the reference path, start from 0.
            idx_ref_traj: Index of the reference trajectory, start from 0.
            ref_path: Global reference path.
            ref_traj: Global reference trajectory.
        """
        self.idx_ref_path = 0
        self.idx_ref_traj = 0
        self.ref_path = ref_path
        if ref_traj is not None:
            self.ref_traj = ref_traj
        else:
            self.ref_traj = self.get_ref_traj(self.ts, ref_path, self.state, self.base_speed)

    def set_ref_states(self, ref_states:np.ndarray=None) -> np.ndarray:
        """Set the local reference states for the coming time step.

        Arguments:
            ref_states: Local (within the horizon) reference states, if not given, generated by the simple `get_ref_states` method.

        Returns:
            ref_states: Each row is a state, the number of rows should be equal to the horizon
        """
        if ref_states is not None:
            self.ref_states = ref_states
        else:
            self.ref_states, self.idx_ref_traj = self.get_ref_states(self.idx_ref_traj, self.ref_traj, self.state, self.N_hor)
        return self.ref_states

    def check_termination_condition(self, state: np.ndarray, action: np.ndarray, final_goal: np.ndarray) -> bool:
        if np.allclose(state[:2], final_goal[:2], atol=0.5, rtol=0) and abs(action[0]) < 0.4:
            terminated = True
            self.idle = True
            if self.vb:
                print(f"{self._prt_name} MPC solution found.")
        else:
            terminated = False
        return terminated


    @staticmethod
    def get_ref_traj(ts: float, ref_path: List[tuple], state: tuple, speed: float) -> List[tuple]:
        """Generate the global reference trajectory from the reference path.

        Returns:
            ref_traj: x, y coordinates and the heading angles
        """
        x, y = state[0], state[1]
        x_next, y_next = ref_path[0][0], ref_path[0][1]
        
        ref_traj = []
        path_idx = 0
        traveling = True
        while(traveling):# for n in range(N):
            while(True):
                dist_to_next = math.hypot(x_next-x, y_next-y)
                if dist_to_next < 1e-9:
                    path_idx += 1
                    x_next, y_next = ref_path[path_idx][0], ref_path[path_idx][1]
                    break
                x_dir = (x_next-x) / dist_to_next
                y_dir = (y_next-y) / dist_to_next
                eta = dist_to_next/speed # estimated time of arrival
                if eta > ts: # move to the target node for t
                    x = x+x_dir*speed*ts
                    y = y+y_dir*speed*ts
                    break # to append the position
                else: # move to the target node then set a new target
                    x = x+x_dir*speed*eta
                    y = y+y_dir*speed*eta
                    path_idx += 1
                    if path_idx > len(ref_path)-1 :
                        traveling = False
                        break
                    else:
                        x_next, y_next = ref_path[path_idx][0], ref_path[path_idx][1]
            if not dist_to_next < 1e-9:
                ref_traj.append((x, y, math.atan2(y_dir,x_dir)))
        return ref_traj
    
    @staticmethod
    def get_ref_states(idx_ref_traj: int, ref_traj: List[tuple], state: tuple, action_steps=1, horizon=20) -> Tuple[np.ndarray, int]:
        """Get the local reference states for MPC.

        Returns:
            ref_states: Each row is a state, the number of rows should be equal to the horizon
            idx_next: The index of the next reference state in the global reference trajectory
        """
        x_ref     = np.array(ref_traj)[:,0].tolist()
        y_ref     = np.array(ref_traj)[:,1].tolist()
        theta_ref = np.array(ref_traj)[:,2].tolist()

        lb_idx = max(0, idx_ref_traj-1*action_steps)                # reduce search space for closest reference point
        ub_idx = min(len(ref_traj), idx_ref_traj+5*action_steps)    # reduce search space for closest reference point

        distances = [math.hypot(state[0]-x[0], state[1]-x[1]) for x in ref_traj[lb_idx:ub_idx]]
        idx_next = distances.index(min(distances))

        idx_next += lb_idx  # idx in orignal reference trajectory list
        if (idx_next+horizon >= len(x_ref)):
            tmpx = x_ref[idx_next:]      + [x_ref[-1]]*(horizon-(len(x_ref)-idx_next))
            tmpy = y_ref[idx_next:]      + [y_ref[-1]]*(horizon-(len(y_ref)-idx_next))
            tmpt = theta_ref[idx_next:]  + [theta_ref[-1]]*(horizon-(len(theta_ref)-idx_next))
        else:
            tmpx = x_ref[idx_next:idx_next+horizon]
            tmpy = y_ref[idx_next:idx_next+horizon]
            tmpt = theta_ref[idx_next:idx_next+horizon]
        ref_states = np.array([tmpx, tmpy, tmpt]).transpose()
        return ref_states, idx_next


    def run_step(self, stc_constraints: list, dyn_constraints: list, other_robot_states:list=None, ref_states:np.ndarray=None, mode:str='safe'):
        """Run the trajectory planner for one step.

        Arguments:
            other_robot_states: A list with length "ns*N_hor*Nother" (E.x. [0,0,0] * (self.N_hor*self.config.Nother))
            ref_states: from "get_ref_states"
        Returns:
            actions: A list of future actions
            pred_states: A list of predicted states
            ref_states: Reference states
            cost: The cost of the predicted trajectory
        """
        ### Mode selection
        # if mode == 'aligning':
        #     if abs(theta_ref[idx_next]-self.state[3])<(math.pi/6):
        #         mode =='work'
        self.set_work_mode(mode)

        if stc_constraints is None:
            stc_constraints = [0] * (self.config.Nstcobs*self.config.nstcobs)
        if dyn_constraints is None:
            dyn_constraints = [0] * (self.config.Ndynobs*self.config.ndynobs*(self.N_hor+1))
        if other_robot_states is None:
            other_robot_states = [0] * (self.ns*(self.N_hor+1)*self.config.Nother)

        ### Get reference states ###
        ref_states = self.set_ref_states(ref_states)
        finish_state = ref_states[-1,:]
        current_refs = ref_states.reshape(-1).tolist()

        ### Get reference velocities ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        if dist_to_goal >= self.base_speed*self.N_hor*self.ts:
            speed_ref_list = [self.base_speed]*self.N_hor
        else:
            speed_ref = dist_to_goal / self.N_hor / self.ts
            speed_ref = max(speed_ref, self.robot_spec.lin_vel_max)
            speed_ref_list = [speed_ref]*self.N_hor

        last_u = self.past_actions[-1] if len(self.past_actions) else np.zeros(self.nu)
            
        ### Assemble parameters for solver & Run MPC###
        params = list(last_u) + list(self.state) + list(finish_state) + self.tuning_params + \
                 current_refs + speed_ref_list + other_robot_states + \
                 stc_constraints + dyn_constraints + self.stc_weights + self.dyn_weights
        try:
            # self.solver_debug(stc_constraints) # use to check (visualize) the environment
            taken_states, pred_states, actions, cost, solver_time, exit_status = self.run_solver(params, self.state, self.config.action_steps)
        except RuntimeError as err:
            if self.use_tcp:
                self.mng.kill()
            print(f"Fatal: Cannot run solver. {err}.")
            return -1

        self.past_states.append(self.state)
        self.past_states += taken_states[:-1]
        self.past_actions += actions
        self.state = taken_states[-1]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)

        if exit_status in self.config.bad_exit_codes and self.vb:
            print(f"{self._prt_name} Bad converge status: {exit_status}")

        return actions, pred_states, ref_states, cost

    def run_solver(self, parameters:list, state: np.ndarray, take_steps:int=1):
        """ Run the solver for the pre-defined MPC problem.

        Arguments:
            parameters: All parameters used by MPC, defined in 'build'.
            state: The current state.
            take_steps: The number of control step taken by the input (default 1).

        Returns:
            taken_states: List of taken states, length equal to take_steps.
            pred_states: List of predicted states at this step, length equal to horizon N.
            actions: List of taken actions, length equal to take_steps.
            cost: The cost value of this step
            solver_time: Time cost for solving MPC of the current time step
            exit_status: The exit state of the solver.

        Comments:
            The motion model (dynamics) is defined initially.
        """
        if self.use_tcp:
            return self.run_solver_tcp(parameters, state, take_steps)

        import opengen as og
        solution:og.opengen.tcp.solver_status.SolverStatus = self.solver.run(parameters)
        
        u           = solution.solution
        cost        = solution.cost
        exit_status = solution.exit_status
        solver_time = solution.solve_time_ms
        
        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model(state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts)
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model(pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts)
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = u[:self.nu*take_steps]
        actions = np.array(actions).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status

    def run_solver_tcp(self, parameters:list, state: np.ndarray, take_steps:int=1):
        solution = self.mng.call(parameters)
        if solution.is_ok():
            # Solver returned a solution
            solution_data = solution.get()
            u           = solution_data.solution
            cost        = solution_data.cost
            exit_status = solution_data.exit_status
            solver_time = solution_data.solve_time_ms
        else:
            # Invocation failed - an error report is returned
            solver_error = solution.get()
            error_code = solver_error.code
            error_msg = solver_error.message
            self.mng.kill() # kill so rust code wont keep running if python crashes
            raise RuntimeError(f"MPC Solver error: [{error_code}]{error_msg}")

        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts )
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model( pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = u[:self.nu*take_steps]
        actions = np.array(actions).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status
    



