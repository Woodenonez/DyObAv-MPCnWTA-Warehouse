import casadi.casadi as cs
from casadi.tools import is_equal

from typing import List, Tuple

import numpy as np
import math
import yaml
import json

from zfilter import KalmanFilter, model_CV
from pkg_dwa_tracker import utils_geo

import numpy as np

class LinearizedCoordinatedTurnModel:
    def __init__(self, dt):
        self.dt = dt

    def predict(self, state, omega):
        x, y, v, phi = state

        F = np.array([
            [1, 0, self.dt * np.cos(phi), -v * self.dt * np.sin(phi)],
            [0, 1, self.dt * np.sin(phi),  v * self.dt * np.cos(phi)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        G = np.array([
            [-v * self.dt * np.sin(phi), v * (np.cos(phi) - np.cos(phi + omega * self.dt)) / omega],
            [ v * self.dt * np.cos(phi), v * (np.sin(phi) - np.sin(phi + omega * self.dt)) / omega],
            [0, 0],
            [0, self.dt]
        ])

        # Linearized state transition
        next_state = np.dot(F, state) + np.dot(G, [0, omega])

        # Return F and G for usage in the Kalman filter
        return next_state, F, G

# Example usage:
lct_model = LinearizedCoordinatedTurnModel(dt=0.1)

state = [0.0, 0.0, 1.0, 0.0]  # Initial state [x, y, v, phi]
omega = 0.1  # Turn rate

# Predict the next state
next_state, F, G = lct_model.predict(state, omega)

