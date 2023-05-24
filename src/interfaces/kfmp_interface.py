import os
import pathlib

import numpy as np

import zfilter
from configs import MpcConfiguration

from typing import List, Tuple


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

class KfmpInterface:
    def __init__(self, config_file_name: str, Q=np.eye(4), R=np.eye(2), state_space:List[np.ndarray]=None):
        self._prt_name = 'KFMPInterface'
        config_file_path = os.path.join(ROOT_DIR, 'config', config_file_name)
        self.config = MpcConfiguration.from_yaml(config_file_path)
        if state_space is None:
            self.state_space = zfilter.model_CV(self.config.ts)
            # self.state_space = zfilter.model_CA(self.config.ts)
        else:
            self.state_space = state_space
        self.kf = zfilter.KalmanFilter(self.state_space, P0=np.eye(4), Q=Q, R=R, pred_offset=self.config.N_hor)

    def get_motion_prediction(self, input_traj: List[tuple], ref_image=None, pred_offset=None, rescale:float=1.0, batch_size=None) -> Tuple[List[tuple], List[float]]:
        """Return a list of predicted positions.
        
        Arguments:
            input_traj: A list of tuples, each tuple is a coordinate (x, y).
            rescale: The scale of the input trajectory.
            ref_image: Placeholder.
            pred_offset: Placeholder.
            batch_size: Placeholder.

        Returns:
            positions: A list of predicted positions.
            uncertainty: A list of uncertainty values.
        """
        if input_traj is None:
            return None
            
        input_traj = [[x*rescale for x in y] for y in input_traj]
        if len(input_traj) > 1:
            init_state = np.array([input_traj[0][0], 
                                input_traj[0][1], 
                                input_traj[1][0] - input_traj[0][0], 
                                input_traj[1][1] - input_traj[0][1], ]).reshape(4,1)
        else:
            init_state = np.array([input_traj[0][0], input_traj[0][1], 0, 0]).reshape(4,1)
        self.kf.set_init_state(init_state)
        _, P = self.kf.inference(np.array(input_traj))
        positions = self.kf.Xs[:2, len(input_traj):].T.tolist()
        uncertainty = [[P[0,0], P[1,1]]]*len(positions)

        return positions, uncertainty