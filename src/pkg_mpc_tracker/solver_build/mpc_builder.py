import casadi.casadi as cs
from opengen import opengen as og # or "import opengen as og"

from . import mpc_helper
from . import mpc_cost

from configs import MpcConfiguration, CircularRobotSpecification

from typing import Union, List, Callable


class MpcModule:
    """Build the MPC module. Define states, inputs, cost, and constraints.

    Functions
        build: Build the MPC problem and solver.
    """
    def __init__(self, mpc_config: MpcConfiguration, robot_config: CircularRobotSpecification):
        self._prt_name = '[MPC-Builder]'
        self.config = mpc_config
        self.robot_config = robot_config
        ### Frequently used
        self.ts = self.config.ts        # sampling time
        self.ns = self.config.ns        # number of states
        self.nu = self.config.nu        # number of inputs
        self.N_hor = self.config.N_hor  # control/pred horizon

    def build(self, motion_model: Callable[[cs.SX, cs.SX, float], cs.SX], use_tcp:bool=False, test:bool=False):
        """Build the MPC problem and solver, including states, inputs, cost, and constraints.

        Arguments:
            motion_model: Callable function `s'=f(s,u,ts)` that generates next state given the current state and action.
            use_tcp : If the solver will be called directly or via TCP.
            test : If the function is called for testing purposes.

        Conmments:
            Inputs (u): speed, angular speed
            states (s): x, y, theta
            
        Reference:
            Ellipse definition: [https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate]
        """
        print(f'{self._prt_name} Building MPC module...')

        u = cs.SX.sym('u', self.nu*self.N_hor)  # 0. Inputs from 0 to N_hor-1

        u_m1 = cs.SX.sym('u_m1', self.nu)       # 1. Input at kt=-1
        s_0 = cs.SX.sym('s_0', self.ns)         # 2. State at kt=0
        s_N = cs.SX.sym('s_N', self.ns)         # 3. State of goal at kt=N_hor
        q = cs.SX.sym('q', self.config.nq)      # 4. Penalty parameters for objective terms
        r_s = cs.SX.sym('r_s', self.ns*self.N_hor)  # 5. Reference states
        r_v = cs.SX.sym('r_v', self.N_hor)          # 6. Reference speed
        c_0 = cs.SX.sym('c_0', self.ns*self.config.Nother)          # 7. States of other robots at kt=0
        c = cs.SX.sym('c', self.ns*self.N_hor*self.config.Nother)   # 8. Predicted states of other robots
        o_s = cs.SX.sym('os', self.config.Nstcobs*self.config.nstcobs)                  # 9. Static obstacles
        o_d = cs.SX.sym('od', self.config.Ndynobs*self.config.ndynobs*(self.N_hor+1))   # 10. Dynamic obstacles
        q_stc = cs.SX.sym('qstc', self.N_hor)               # 11. Static obstacle weights
        q_dyn = cs.SX.sym('qdyn', self.N_hor)               # 12. Dynamic obstacle weights

        z = cs.vertcat(u_m1, s_0, s_N, q, r_s, r_v, c_0, c, o_s, o_d, q_stc, q_dyn)
        
        (x, y, theta) = (s_0[0], s_0[1], s_0[2])
        (x_goal, y_goal, theta_goal) = (s_N[0], s_N[1], s_N[2])
        (v_init, w_init) = (u_m1[0], u_m1[1])
        (qpos, qvel, qtheta, rv, rw)                    = (q[0], q[1], q[2], q[3], q[4])
        (qN, qthetaN, qrpd, acc_penalty, w_acc_penalty) = (q[5], q[6], q[7], q[8], q[9])
        
        ref_states = cs.reshape(r_s, (self.ns, self.N_hor)).T
        ref_states = cs.vertcat(ref_states, ref_states[[-1],:])[:, :2]

        cost = 0
        penalty_constraints = 0
        state = cs.vcat([x,y,theta])
        for kt in range(0, self.N_hor): # LOOP OVER PREDICTIVE HORIZON
            
            ### Run step with motion model
            u_t = u[kt*self.nu:(kt+1)*self.nu]  # inputs at time t
            state = motion_model(state, u_t, self.ts) # Kinematic/dynamic model

            ### Reference deviation costs
            cost += mpc_cost.cost_refpath_deviation(state.T, ref_states[kt:, :], weight=qrpd)   # [cost] reference path deviation cost
            cost += mpc_cost.cost_refvalue_deviation(u_t[0], r_v[kt], weight=qvel)              # [cost] refenrence velocity deviation
            cost += mpc_cost.cost_control_actions(u_t.T, weights=cs.horzcat(rv, rw))            # [cost] penalize control actions

            ### Fleet collision avoidance
            other_x_0 = c_0[self.ns  ::self.ns] # first  state
            other_y_0 = c_0[self.ns+1::self.ns] # second state
            other_robots_0 = cs.hcat([other_x_0, other_y_0]) # states of other robots at time 0
            other_robots_0 = cs.transpose(other_robots_0) # every column is a state of a robot
            cost += mpc_cost.cost_fleet_collision(state[:2].T, other_robots_0.T, safe_distance=self.robot_config.vehicle_width, weight=1000)

            ### Fleet collision avoidance [Predictive]
            other_robots_x = c[kt*self.ns  ::self.ns*self.N_hor] # first  state
            other_robots_y = c[kt*self.ns+1::self.ns*self.N_hor] # second state
            other_robots = cs.hcat([other_robots_x, other_robots_y]) # states of other robots at time kt (Nother*ns)
            other_robots = cs.transpose(other_robots) # every column is a state of a robot
            cost += mpc_cost.cost_fleet_collision(state[:2].T, other_robots.T, safe_distance=self.robot_config.vehicle_width, weight=10)

            ### Static obstacles
            for i in range(self.config.Nstcobs):
                eq_param = o_s[i*self.config.nstcobs : (i+1)*self.config.nstcobs]
                n_edges = int(self.config.nstcobs / 3) # 3 means b, a0, a1
                b, a0, a1 = eq_param[:n_edges], eq_param[n_edges:2*n_edges], eq_param[2*n_edges:]

                inside_stc_obstacle = mpc_helper.inside_cvx_polygon(state.T, b.T, a0.T, a1.T)
                penalty_constraints += cs.fmax(0, cs.vertcat(inside_stc_obstacle))

                cost += mpc_cost.cost_inside_cvx_polygon(state.T, b.T, a0.T, a1.T, weight=q_stc[kt])

            ### Dynamic obstacles
            x_dyn     = o_d[0::self.config.ndynobs*(self.N_hor+1)]
            y_dyn     = o_d[1::self.config.ndynobs*(self.N_hor+1)]
            rx_dyn    = o_d[2::self.config.ndynobs*(self.N_hor+1)]
            ry_dyn    = o_d[3::self.config.ndynobs*(self.N_hor+1)]
            As        = o_d[4::self.config.ndynobs*(self.N_hor+1)]
            alpha_dyn = o_d[5::self.config.ndynobs*(self.N_hor+1)]

            inside_dyn_obstacle = mpc_helper.inside_ellipses(state.T, [x_dyn, y_dyn, rx_dyn, ry_dyn, As])
            penalty_constraints += cs.fmax(0, inside_dyn_obstacle)

            ellipse_param = [x_dyn, y_dyn, 
                             rx_dyn+self.robot_config.vehicle_margin+self.robot_config.social_margin, 
                             ry_dyn+self.robot_config.vehicle_margin+self.robot_config.social_margin, 
                             As, alpha_dyn]
            cost += mpc_cost.cost_inside_ellipses(state.T, ellipse_param, weight=1000)

            ### Dynamic obstacles [Predictive]
            # (x, y, rx, ry, angle, alpha) for obstacle 0 for N_hor steps, then similar for obstalce 1 for N_hor steps...
            x_dyn     = o_d[(kt+1)*self.config.ndynobs  ::self.config.ndynobs*(self.N_hor+1)]
            y_dyn     = o_d[(kt+1)*self.config.ndynobs+1::self.config.ndynobs*(self.N_hor+1)]
            rx_dyn    = o_d[(kt+1)*self.config.ndynobs+2::self.config.ndynobs*(self.N_hor+1)]
            ry_dyn    = o_d[(kt+1)*self.config.ndynobs+3::self.config.ndynobs*(self.N_hor+1)]
            As        = o_d[(kt+1)*self.config.ndynobs+4::self.config.ndynobs*(self.N_hor+1)]
            alpha_dyn = o_d[(kt+1)*self.config.ndynobs+5::self.config.ndynobs*(self.N_hor+1)]

            inside_dyn_obstacle = mpc_helper.inside_ellipses(state.T, [x_dyn, y_dyn, rx_dyn, ry_dyn, As])
            penalty_constraints += cs.fmax(0, inside_dyn_obstacle)

            ellipse_param = [x_dyn, y_dyn, 
                             rx_dyn+self.robot_config.vehicle_margin, 
                             ry_dyn+self.robot_config.vehicle_margin, 
                             As, alpha_dyn]
            cost += mpc_cost.cost_inside_ellipses(state.T, ellipse_param, weight=q_dyn[kt])
        
        ### Terminal cost
        # state_final_goal = cs.vertcat(x_goal, y_goal, theta_goal)
        # cost += mpc_cost.cost_refstate_deviation(state, state_final_goal, weights=cs.vertcat(qN, qN, qthetaN)) 
        cost += qN*((state[0]-x_goal)**2 + (state[1]-y_goal)**2) + qthetaN*(state[2]-theta_goal)**2 # terminated cost

        ### Max speed bound
        umin = [self.robot_config.lin_vel_min, -self.robot_config.ang_vel_max] * self.N_hor
        umax = [self.robot_config.lin_vel_max,  self.robot_config.ang_vel_max] * self.N_hor
        bounds = og.constraints.Rectangle(umin, umax)

        ### Acceleration bounds and cost
        v = u[0::2] # velocity
        w = u[1::2] # angular velocity
        acc   = (v-cs.vertcat(v_init, v[0:-1]))/self.ts
        w_acc = (w-cs.vertcat(w_init, w[0:-1]))/self.ts
        acc_constraints = cs.vertcat(acc, w_acc)
        # Acceleration bounds
        acc_min   = [ self.robot_config.lin_acc_min] * self.N_hor 
        w_acc_min = [-self.robot_config.ang_acc_max] * self.N_hor
        acc_max   = [ self.robot_config.lin_acc_max] * self.N_hor
        w_acc_max = [ self.robot_config.ang_acc_max] * self.N_hor
        acc_bounds = og.constraints.Rectangle(acc_min + w_acc_min, acc_max + w_acc_max)
        # Accelerations cost
        cost += cs.mtimes(acc.T, acc)*acc_penalty
        cost += cs.mtimes(w_acc.T, w_acc)*w_acc_penalty

        problem = og.builder.Problem(u, z, cost) \
            .with_constraints(bounds) \
            .with_aug_lagrangian_constraints(acc_constraints, acc_bounds)
        problem.with_penalty_constraints(penalty_constraints)

        build_config = og.config.BuildConfiguration() \
            .with_build_directory(self.config.build_directory) \
            .with_build_mode(self.config.build_type)
        if not use_tcp:
            build_config.with_build_python_bindings()
        else:
            build_config.with_tcp_interface_config()

        meta = og.config.OptimizerMeta() \
            .with_optimizer_name(self.config.optimizer_name)

        solver_config = og.config.SolverConfiguration() \
            .with_initial_penalty(10) \
            .with_max_duration_micros(self.config.max_solver_time)
            # initial penalty = 1
            # tolerance = 1e-4
            # max_inner_iterations = 500 (given a penalty factor)
            # max_outer_iterations = 10  (increase the penalty factor)
            # penalty_weight_update_factor = 5.0
            # max_duration_micros = 5_000_000 (5 sec)

        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config) \
            .with_verbosity_level(1)
        if test:
            print(f"{self._prt_name} MPC builder is tested without building.")
            return 1
        else:
            builder.build()

        print(f'{self._prt_name} MPC module built.')

