import pathlib
from typing import List

import numpy as np
import torch
import torchvision

import pkg_motion_prediction.pre_load as pre_load
from pkg_motion_prediction.utils import utils_np
from pkg_motion_prediction.data_handle import data_handler

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

class MmpInterface:
    def __init__(self, config_file_name: str):
        self._prt_name = 'MMPInterface'
        self.config = pre_load.load_config(ROOT_DIR, config_file_name)
        self.network_manager = pre_load.load_net(ROOT_DIR, config_file_name)

    def get_motion_prediction(self, input_traj: List[tuple], ref_image: torch.Tensor, pred_offset: int, rescale:float=1.0, batch_size:int=1) -> List[np.ndarray]:
        """Return a list of predicted positions.
        
        Arguments:
            input_traj: A list of tuples, each tuple is a coordinate (x, y).
            ref_image: A tensor of shape (height, width).
            pred_offset: The maximum number of steps to predict.
            rescale: The scale of the input trajectory.
            batch_size: The batch size for inference.

        Returns:
            A list (length is horizon) of predicted positions, each element is a ndarry with K (#hypo) rows.
        """
        if input_traj is None:
            return None
            
        input_traj = [[x*rescale for x in y] for y in input_traj]
        transform = torchvision.transforms.Compose([data_handler.ToTensor()])
        if not isinstance(ref_image, torch.Tensor):
            raise TypeError(f'The reference image should be a tensor, got {type(ref_image)}.')
        input_ = pre_load.traj_to_input(input_traj, ref_image=ref_image, transform=transform, obsv_len=self.config.obsv_len)
        
        hypos_list:List[np.ndarray] = []

        ### Batch inference
        input_all = input_.unsqueeze(0)
        for offset in range(1, pred_offset+1):
            input_[-1,:,:] = offset*torch.ones_like(input_[-1,:,:])
            input_all = torch.cat((input_all, input_.unsqueeze(0)), dim=0)
        input_all = input_all[1:]
        for i in range(pred_offset//batch_size):
            input_batch = input_all[batch_size*i:batch_size*(i+1), :]
            try:
                hyposM = torch.concat((hyposM, self.network_manager.inference(input_batch)), dim=0)
            except:
                hyposM = self.network_manager.inference(input_batch)
        if pred_offset%batch_size > 0:
            input_batch = input_all[batch_size*(pred_offset//batch_size):, :]
            hyposM = torch.concat((hyposM, self.network_manager.inference(input_batch)), dim=0)
        for i in range(pred_offset):
            hypos_list.append(utils_np.get_closest_edge_point(hyposM[i,:].numpy(), 255 - ref_image.numpy()) / rescale)

        ### Single inference
        # for offset in range(1, pred_offset+1):
        #     input_[-1,:,:] = offset*torch.ones_like(input_[-1,:,:])

        #     hyposM = self.net.inference(input_.unsqueeze(0))[0,:]

        #     hyposM = utils_np.get_closest_edge_point(hyposM, 255 - ref_image) # post-processing
        #     hypos_list.append(hyposM/rescale)
        return hypos_list