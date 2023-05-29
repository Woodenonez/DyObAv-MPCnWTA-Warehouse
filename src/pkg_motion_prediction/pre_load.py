import os
import sys
import time

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from .network_manager import NetworkManager
from .net_module.net import ConvMultiHypoNet
from .data_handle.data_handler import DataHandler
from .data_handle.dataset import ImageStackDataset

from .utils import utils_yaml
from .utils import utils_np

import pickle
from datetime import datetime

from configs import WtaNetConfiguration
from typing import Tuple

def check_device():
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count(),
              'Current 1:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
    else:
        print(f'CUDA not working! Pytorch: {torch.__version__}.')
        sys.exit(0)
    torch.cuda.empty_cache()
    
def load_config(root_dir: str, config_file: str) -> WtaNetConfiguration:
    yaml_path = os.path.join(root_dir, 'config/', config_file)
    return WtaNetConfiguration.from_yaml(yaml_path, with_partition=True)

def load_path(config: WtaNetConfiguration):
    """Return `model_path`, `label_path`, `data_path`."""
    return config.model_path, config.label_path, config.data_path

def load_data(config: WtaNetConfiguration, paths: Tuple[str], transform: torchvision.transforms, 
              validation_cache=10, load_for_test=False, num_workers=0):
    """Load dataset and data handler."""
    myDS = ImageStackDataset(csv_path=paths[1], root_dir=paths[2], transform=transform, 
                   dynamic_env=config.dynamic_env, pred_traj=(config.pred_len>1), T_channel=True)
    myDH = DataHandler(myDS, batch_size=config.batch_size, validation_cache=validation_cache, num_workers=num_workers)
    if not load_for_test:
        print("Data prepared. #Samples(training, val):{}, #Batches:{}".format(myDH.return_length_ds(), myDH.return_length_dl()))
    print('Sample: {\'image\':',myDS[0]['image'].shape,'\'label\':',myDS[0]['label'],'}')
    return myDS, myDH

def load_manager(config: WtaNetConfiguration, loss: dict) -> NetworkManager:
    net = ConvMultiHypoNet(config.input_channel, config.dim_out, config.fc_input, num_components=config.num_hypos)
    net_manager = NetworkManager(config, net, loss)
    net_manager.build_Network()
    return net_manager

def save_profile(manager: NetworkManager, save_path='./'):
    now = datetime.now()
    dt = now.strftime("%d_%m_%Y__%H_%M_%S")

    manager.plot_history_loss()
    plt.savefig(os.path.join(save_path, dt+'.png'), bbox_inches='tight')
    plt.close()

    loss_dict = {'loss':manager.Loss, 'val_loss':manager.Val_loss}
    with open(os.path.join(save_path, dt+'.pickle'), 'wb') as pf:
        pickle.dump(loss_dict, pf)

def main_train(root_dir, config_file, transform, loss, k_top_list, num_workers:int):
    ### Check and load
    check_device()
    config = load_config(root_dir, config_file)
    paths = load_path(config)
    _, myDH = load_data(config, paths, transform, num_workers=num_workers)
    network_manager = load_manager(config, loss)
    network_manager.build_Network()

    ### Training
    start_time = time.time()
    network_manager.train(myDH, config.batch_size, config.epoch, k_top_list=k_top_list, val_after_batch=config.batch_size)
    total_time = round((time.time()-start_time)/3600, 4)
    if (paths[0] is not None) & network_manager.complete:
        torch.save(network_manager.model.state_dict(), paths[0])
    nparams = sum(p.numel() for p in network_manager.model.parameters() if p.requires_grad)
    print("\nTraining done: {} parameters. Cost time: {}h.".format(nparams, total_time))

    save_profile(network_manager)

def load_net(root_dir, config_file) -> NetworkManager:
    config = load_config(root_dir, config_file)
    paths = load_path(config)
    network_manager = load_manager(config, {})
    network_manager.build_Network()
    network_manager.model.load_state_dict(torch.load(paths[0]))
    network_manager.model.eval() # with BN layer, must run eval first
    return network_manager

def main_test_pre(root_dir, config_file, transform):
    ### Check and load
    config = load_config(root_dir, config_file)
    paths = load_path(config)
    myDS, myDH = load_data(config, paths, transform, load_for_test=True)
    network_manager = load_net(root_dir, config_file)
    return myDS, myDH, network_manager

def main_test(dataset: ImageStackDataset, network_manager: NetworkManager, idx: int, offset=None):
    img   = dataset[idx]['image']
    label = dataset[idx]['label']
    traj  = dataset[idx]['traj']
    index = dataset[idx]['index']
    if offset is not None:
        img[-1,:,:] = offset*torch.ones_like(img[-1,:,:])
    pred = network_manager.inference(img.unsqueeze(0))
    reference = dataset[idx]['reference']
    return img, label, traj, index, pred, reference

def traj_to_input(traj:list, ref_image:np.array, transform=None, obsv_len:int=5) -> torch.Tensor:
    if len(traj)<obsv_len:
        traj = traj + [traj[-1]]*(obsv_len-len(traj))
    traj = traj[-obsv_len:]

    input_img = np.empty(shape=[ref_image.shape[0],ref_image.shape[1],0])
    for position in traj:
        this_x, this_y = position[0], position[1]

        obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(ref_image), sigmas=[20,20])
        input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
    input_img = np.concatenate((input_img, ref_image[:,:,np.newaxis]), axis=2)
    input_img = np.concatenate((input_img, ref_image[:,:,np.newaxis]), axis=2) # placeholder for the pred_offset channel

    sample = {'image':input_img, 'label':0}
    if transform:
        sample = transform(sample)
    return sample['image']


