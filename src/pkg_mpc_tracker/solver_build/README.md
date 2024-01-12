## Description
This folder contains everything to build an MPC solver.

## Dependencies
```
casadi, opengen
```

## List of Contents
- `unit_test.py`: Unit test for the MPC solver.
- `mpc_builder.py`: The main script to build the MPC solver.
- `mpc_cost.py`: Cost terms for the MPC solver.
- `mpc_helper.py`: Helper functions for the MPC solver.

## Dependency Chain
(A->B means B depends on A.)
```
mpc_helper.py -> mpc_cost.py -> mpc_builder.py
```

## MPC problem
For the current time step $k=0$, given the horizon $N$ and the initial state $x_0$, the MPC problem is to find the optimal control sequence $u_0, u_1, \dots, u_{N-1}$ that minimizes the cost function.