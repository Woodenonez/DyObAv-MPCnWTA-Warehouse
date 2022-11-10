import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, random_split

'''
'''

class DataHandler():
    def __init__(self, dataset, batch_size=64, shuffle=True, validation_prop=0.2, validation_cache=64, num_workers=0):
        self.__val_p = validation_prop
        self.dataset = dataset
        if 0<validation_prop<1:
            self.split_dataset()
        else:
            self.dataset_train = self.dataset
            self.dataset_val = []
            self.dl_val = []

        self.dl = DataLoader(self.dataset_train, batch_size, shuffle, num_workers=num_workers) # create the dataloader from the dataset
        self.__iter = iter(self.dl)

        if self.dataset_val:
            self.dl_val = DataLoader(self.dataset_val, batch_size=validation_cache, shuffle=shuffle, num_workers=num_workers)
            self.__iter_val = iter(self.dl_val)

    def split_dataset(self):
        ntraining = int(self.return_length_ds(whole_dataset=True) * (1-self.__val_p))
        nval = self.return_length_ds(whole_dataset=True) - ntraining
        self.dataset_train, self.dataset_val = random_split(self.dataset, [ntraining, nval])

    def return_batch(self):
        try:
            sample_batch = next(self.__iter)
        except StopIteration:
            self.reset_iter()
            sample_batch = next(self.__iter)
        return sample_batch['image'], sample_batch['label']

    def return_val(self):
        try:
            sample_batch = next(self.__iter_val)
        except StopIteration:
            self.__iter_val = iter(self.dl_val)
            sample_batch = next(self.__iter_val)
        image, label = sample_batch['image'], sample_batch['label']
        if len(image.shape)==3:
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
        return image, label

    def reset_iter(self):
        self.__iter = iter(self.dl)

    def return_length_ds(self, whole_dataset=False):
        if whole_dataset:
            return len(self.dataset)
        else:
            return len(self.dataset_train), len(self.dataset_val)

    def return_length_dl(self):
        return len(self.dl) # the number of batches, only for training dataset


class Rescale(object):
    def __init__(self, output_size:tuple, tolabel=False):
        '''
        Args:
            output_size - (height * width)
        '''
        super().__init__()
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        self.tolabel = tolabel

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        h_new, w_new = self.output_size
        if (h==h_new) & (w==w_new): # if no need to resize, skip
            return sample

        # img = skimage.transform.resize(image, (h_new,w_new))
        img = torchvision.transforms.Resize((h_new,w_new))(image)
        if self.tolabel:
            label = [(x[0]*w_new/w, x[1]*h_new/h) for x in label]
        return {'image':img, 'label':label}

class ToGray(object):
    # For RGB the weight could be (0.299R, 0.587G, 0.114B)
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            assert (len(weight)==3)
            w1 = round(weight[0]/sum(weight),3)
            w2 = round(weight[1]/sum(weight),3)
            w3 = 1 - w1 - w2
            self.weight = (w1, w2, w3)
        else:
            self.weight = None

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if (len(image.shape)==2) or (image.shape[2] == 1):
            return sample
        else:
            image = image[:,:,:3] # ignore alpha
            if self.weight is not None:
                img = self.weight[0]*image[:,:,0] + self.weight[1]*image[:,:,1] + self.weight[2]*image[:,:,2]
            else:
                img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
        return {'image':img[:,np.newaxis], 'label':label}

class DelAlpha(object):
    # From RGBA to RGB
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if (len(image.shape)==2) or (image.shape[2] == 1):
            return sample
        else:
            return {'image':image[:,:,:3], 'label':label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        label = np.array(label)
        # swap color axis, numpy: H x W x C -> torch: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class MaxNormalize(object):
    def __init__(self, max_pixel=255, max_label=10):
        super().__init__()
        self.mp = max_pixel
        self.ml = max_label

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if not isinstance(self.mp, (tuple,list)):
            self.mp = [self.mp]*image.shape[2]
        for i in range(image.shape[2]):
            image[:,:,:i] = image[:,:,:i]/self.mp[i]
        if self.ml is not None:
            if self.ml is tuple:
                label = [(x[0]/self.ml[0], x[1]/self.ml[1]) for x in label]
            else:
                label = [(x[0]/self.ml, x[1]/self.ml) for x in label]
        return {'image':image, 'label':label}

