from typing import Union

import numpy as np
import casadi.casadi as cs

from util.basic_datatype import *

#%%## Define the kinematics here ###
def kinematics_ct(ts:SamplingTime, x:Union[NumpyState, CasadiState], u:Union[NumpyAction, CasadiAction]) -> Union[NumpyState, CasadiState]: 
    # δ(state) per ts
    if isinstance(x, np.ndarray):
        dx_x     = ts * (u[0]*np.cos(x[2]))
        dx_y     = ts * (u[0]*np.sin(x[2]))
        dx_theta = ts * u[1]
        return np.array([dx_x, dx_y, dx_theta])
    elif isinstance(x, cs.SX):
        dx_x     = ts * (u[0]*cs.cos(x[2]))
        dx_y     = ts * (u[0]*cs.sin(x[2]))
        dx_theta = ts * u[1]
        return cs.vertcat(dx_x, dx_y, dx_theta)
    else:
        raise TypeError(f'The input should be "numpy.ndarray" or "casadi.SX", got {type(x)}.')

def kinematics_rk1(ts:SamplingTime, x:Union[NumpyState, CasadiState], u:Union[NumpyAction, CasadiAction]) -> Union[NumpyState, CasadiState]: 
    # discretized via Runge-Kutta 1 (Euler method)
    return x + kinematics_ct(ts, x, u)

def kinematics_rk4(ts:SamplingTime, x:Union[NumpyState, CasadiState], u:Union[NumpyAction, CasadiAction]) -> Union[NumpyState, CasadiState]: 
    # discretized via Runge-Kutta 4
    k1 = kinematics_ct(ts, x, u)
    k2 = kinematics_ct(ts, x + 0.5*k1, u)
    k3 = kinematics_ct(ts, x + 0.5*k2, u)
    k4 = kinematics_ct(ts, x + k3, u)
    x_next = x + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

def kinematics_simple(ts:SamplingTime, x:Union[NumpyState, CasadiState], u:Union[NumpyAction, CasadiAction]) -> Union[NumpyState, CasadiState]: 
    return x + ts*u


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    point = cs.DM([1,0])
    ellipse = cs.DM([0,0,1,1,0])

    weight = 1
    narrowness = 5
    x = np.linspace(-5, 5, 100)
    y = weight / (1+np.exp(-narrowness*x-4))

    plt.plot(x, y)
    plt.show()

    import sys
    sys.exit(0)
    
    C  = np.array([2, 1]).reshape(-1,1)
    U0 = np.array([1, 1]).reshape(-1,1)
    U1 = np.array([-1, 1]).reshape(-1,1)
    e0, e1 = 2, 1
    a0 = np.arange(start=-e0, stop=e0, step=0.01)
    a1 = np.concatenate([np.sqrt(1-(a0/e0)**2)*e1, -np.sqrt(1-(a0/e0)**2)*e1])
    a0 = np.tile(a0, 2)

    P = C + a0*U0 + a1*U1

    plt.plot(P[0,:], P[1,:], '.')
    plt.plot(C[0], C[1], 'rx')
    plt.axis('equal')
    plt.show()
