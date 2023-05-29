import os
import pathlib

import numpy as np

import zfilter
from configs import MpcConfiguration

from typing import List, Tuple


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

class CvmpInterface:
    def __init__(self, config_file_name: str, state_space:List[np.ndarray]=None):
        self._prt_name = 'KFMPInterface'
        config_file_path = os.path.join(ROOT_DIR, 'config', config_file_name)
        self.config = MpcConfiguration.from_yaml(config_file_path)
        if state_space is None:
            self.state_space = zfilter.model_CV(self.config.ts)
        else:
            self.state_space = state_space

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
        
        input_traj = input_traj[-5:] if len(input_traj) > 5 else input_traj # only use the latest 5 points
            
        input_traj = [[x*rescale for x in y] for y in input_traj]
        if len(input_traj) > 1:
            vx_list = [input_traj[i+1][0] - input_traj[i][0] for i in range(len(input_traj)-1)]
            vy_list = [input_traj[i+1][1] - input_traj[i][1] for i in range(len(input_traj)-1)]
            vx = np.mean(vx_list)
            vy = np.mean(vy_list)
        else:
            vx = 0
            vy = 0

        positions = []
        for i in range(self.config.N_hor):
            positions.append([input_traj[-1][0] + vx*(i+1), input_traj[-1][1] + vy*(i+1)])
        uncertainty = [[1.0, 1.0]]*len(positions)

        return positions, uncertainty