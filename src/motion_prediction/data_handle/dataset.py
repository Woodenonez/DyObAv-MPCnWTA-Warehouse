import os
import glob

import numpy as np
import pandas as pd
# from skimage import io
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset

from motion_prediction.util import utils_np

'''
'''
class io():
    @staticmethod
    def imread(path):
        tr = torchvision.transforms.PILToTensor()
        return tr(Image.open(path)).permute(1,2,0)


class ImageStackDataset(Dataset):
    def __init__(self, csv_path:str, root_dir:str, transform:torchvision.transforms=None, 
                 T_channel:bool=False, dynamic_env:bool=False, pred_traj:bool=False):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform

        self.with_T = T_channel
        self.dyn_env = dynamic_env
        self.pred_traj = pred_traj

        self.ext = 'jpg'  # should not have '.'
        self.csv_str = 'p' # in csv files, 'p' means position

        self.input_len = len([x for x in list(self.info_frame) if self.csv_str in x]) # length of input time step
        self.img_shape = self.check_img_shape()

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        index = info['index']
        traj = []
        if self.dyn_env:
            for i in range(self.input_len):
                img_name = info['p{}'.format(i)].split('_')[-1] + '.' + self.ext
                img_path = os.path.join(self.root_dir, str(index), img_name)
                this_x = float(info['p{}'.format(i)].split('_')[0])
                this_y = float(info['p{}'.format(i)].split('_')[1])
                traj.append([this_x,this_y])

                image = self.togray(io.imread(img_path))
                obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[20,20])
                input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis], image[:,:,np.newaxis]), axis=2)
        else:
            # img_name = f'{info["index"]}.{self.ext}'
            img_name = 'label.png' # XXX
            img_path = os.path.join(self.root_dir, str(int(index)), img_name)
            image = self.togray(io.imread(img_path))
            for i in range(self.input_len):
                position = info['p{}'.format(i)]
                this_x = float(position.split('_')[0])
                this_y = float(position.split('_')[1])
                traj.append([this_x,this_y])

                obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[20,20])
                input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
            input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)
        
        if self.pred_traj:
            label_name_list = [x for x in list(self.info_frame) if 'T' in x]
            label_list = list(info[label_name_list].values)
            label = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]
        else:
            label = [(float(info['T'].split('_')[0]), float(info['T'].split('_')[1]))]

        if self.with_T:
            try:
                offset = int(info['T'].split('_')[2])
            except:
                offset = int(1)
            T_channel = np.ones(shape=[self.img_shape[0],self.img_shape[1],1])*offset # T_channel
            input_img = np.concatenate((input_img, T_channel), axis=2)                                      # T_channel

        sample = {'image':input_img, 'label':label}
        if self.tr:
            sample = self.tr(sample)
        sample['index'] = index
        sample['traj'] = traj
        sample['time'] = info['t']
        sample['reference'] = io.imread(img_path)

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    @staticmethod
    def togray(image):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            return img

    def check_img_shape(self, img_name='label.png'):
        info = self.info_frame.iloc[0]
        if img_name is None:
            img_name = str(info['t']) + '.' + self.ext
        video_folder = str(int(info['index']))
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape

    