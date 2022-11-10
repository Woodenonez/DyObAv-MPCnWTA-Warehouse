import os, sys
from pathlib import Path

import torchvision

from motion_prediction.net_module import loss_functions as loss_func
from motion_prediction.net_module.net import ConvMultiHypoNet
from motion_prediction.data_handle import data_handler as dh
from motion_prediction.data_handle import dataset as ds

import motion_prediction.pre_load as pre_load

print("Program: training\n")

TRAIN_MODEL = 'swta' # ewta, awta, swta

### Config
root_dir = Path(__file__).parents[1]
config_file = 'wsd_1t20_train.yml'

if TRAIN_MODEL == 'ewta':
    meta_loss = loss_func.meta_loss
    k_top_list = [20]*10 + [10]*10 + [8]*10 + [7]*10 + [6]*10 + [5]*10 + [4]*10 + [3]*10 + [2]*10 + [1]*10
elif TRAIN_MODEL == 'awta':
    meta_loss = loss_func.ameta_loss
    k_top_list = [20]*10 + [10]*10 + [8]*10 + [7]*10 + [6]*10 + [5]*10 + [4]*10 + [3]*10 + [2]*10 + [1]*10
elif TRAIN_MODEL == 'swta':
    meta_loss = loss_func.ameta_loss
    k_top_list = [20]*2 + [10]*2 + [8]*1 + [7]*1 + [6]*1 + [5]*2 + [4]*1 + [3]*2 + [2]*2 + [1]*3 + [0]*3
else:
    raise ModuleNotFoundError(f'Cannot find mode {TRAIN_MODEL}.')

# k_top_list = [20]*10 + [10]*10 + [8]*10 + [7]*10 + [6]*10 + [5]*10 + [4]*10 + [3]*10 + [2]*10 + [1]*10

loss_dict = {'meta':meta_loss, 'base':loss_func.loss_mse, 'metric':None}

composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = ConvMultiHypoNet

### Training
pre_load.main_train(root_dir, config_file, Dataset=Dataset, Net=Net, 
                    transform=composed, loss=loss_dict, k_top_list=k_top_list, num_workers=2)
