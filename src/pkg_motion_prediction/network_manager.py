import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from timeit import default_timer as timer
from datetime import timedelta

from configs import WtaNetConfiguration
from .data_handle.data_handler import DataHandler

from typing import List


class NetworkManager():
    """Network Manager is in charge of the training and validation of a neural network."""

    def __init__(self, config: WtaNetConfiguration, net:nn.Module, loss_function_dict: dict, verbose=True):
        """Initialize the Network Manager.
        
        Arguments:
            net: The neural network to be trained, should have attribute "M" for number of components.
            loss_function_dict: A dictionary of loss functions, should have keys "meta", "base", "metric".
        """
        assert(isinstance(loss_function_dict, dict)),('The "loss_function_dict" should be a dict.')
        self._prt_name = "NetManager"
        self.vb = verbose
        self._load_parameters(config)
        self._training_time(None, None, None, init=True)
        if not verbose:
            warnings.filterwarnings("ignore", category=UserWarning)


        self.Loss = []      # track the loss
        self.Oracle_valloss = [] # track the closest component's loss
        self.Val_loss= []   # track the validation loss

        self.net = net
        self.M = net.M # number of components
        try:
            self.loss_meta = loss_function_dict['meta']
            self.loss_base = loss_function_dict['base']
            self.metric = loss_function_dict['metric']
        except:
            warnings.warn(">>> No loss function detected <<<", UserWarning)

        self.complete = False

    def _load_parameters(self, config: WtaNetConfiguration):
        self.lr = config.learning_rate
        self.wr = config.weight_regularization
        self.es = config.early_stopping
        self.cp = config.checkpoint_dir

        self.device:str = config.device
        if (self.device.lower() in ['cuda', 'multi']) and (not torch.cuda.is_available()):
            warnings.warn('>>> CUDA is not available, using CPU instead <<<', UserWarning)
            self.device = 'cpu'

    def _training_time(self, remaining_epoch, remaining_batch, batch_per_epoch, init=False):
        if init:
            self.batch_time = []
            self.epoch_time = []
        else:
            batch_time_average = sum(self.batch_time)/max(len(self.batch_time),1)
            if len(self.epoch_time) == 0:
                epoch_time_average = batch_time_average * batch_per_epoch
            else:
                epoch_time_average = sum(self.epoch_time)/max(len(self.epoch_time),1)
            eta = round(epoch_time_average * remaining_epoch + batch_time_average * remaining_batch, 0)
            return timedelta(seconds=batch_time_average), timedelta(seconds=epoch_time_average), timedelta(seconds=eta)

    def build_Network(self):
        self.gen_Model()
        self.gen_Optimizer(self.model.parameters())

    def gen_Model(self):
        self.model = nn.Sequential()
        self.model.add_module('Net', self.net)
        if self.device == 'multi':
            self.model = nn.DataParallel(self.model.to(torch.device("cuda:0")))
        elif self.device == 'cuda':
            self.model = self.model.to(torch.device("cuda:0"))
        elif self.device == 'cpu': 
            pass
        else:
            raise ModuleNotFoundError(f'No such device as {self.device} (should be "multi", "cuda", or "cpu").')
        return self.model

    def gen_Optimizer(self, parameters):
        self.optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.wr, betas=(0.99, 0.999))
        # self.optimizer = optim.SGD(parameters, lr=1e-3, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        return self.optimizer

    def inference(self, input_data: torch.Tensor) -> torch.Tensor:
        """Inference the model on the input data.
        
        Returns:
            hyposM: The output of the model, reshaped to BxMxC.
        """
        if self.device in ['multi', 'cuda']:
            device = torch.device("cuda")
        else:
            device = 'cpu'
        with torch.no_grad():
            hypos:torch.Tensor = self.model(input_data.float().to(device)).cpu()
        hyposM = hypos.reshape(hypos.shape[0], self.M, -1) # BxMxC
        return hyposM

    def validate(self, data, labels, loss_function, k_top=1):
        outputs = self.model(data)
        loss = self.loss_meta(outputs, self.M, labels, loss_function, k_top=k_top)
        return loss

    def train_batch(self, batch, label, loss_function, k_top=1):
        self.model.zero_grad()
        loss = self.validate(batch, label, loss_function, k_top)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, data_handler: DataHandler, batch_size: int, epoch: int, k_top_list: List[int], val_after_batch=1):
        print('\nTraining...')
        if self.device in ['multi', 'cuda']:
            device = torch.device("cuda")
        else:
            device = 'cpu'

        data_val = data_handler.dataset_val
        max_cnt_per_epoch = data_handler.return_length_dl()
        min_val_loss = np.Inf
        min_val_loss_epoch = np.Inf
        epochs_no_improve = 0
        cnt = 0 # counter for batches over all epochs
        for ep in range(epoch):
            epoch_time_start = timer() ### TIMER

            cnt_per_epoch = 0 # counter for batches within the epoch

            k_top = k_top_list[ep]
            loss_epoch = self.loss_base

            while (cnt_per_epoch<max_cnt_per_epoch):
                cnt += 1
                cnt_per_epoch += 1

                batch_time_start = timer() ### TIMER

                batch, label = data_handler.return_batch()
                batch, label = batch.float().to(device), label.float().to(device)

                loss = self.train_batch(batch, label, loss_function=loss_epoch, k_top=k_top) # train here
                self.Loss.append(loss.item())

                if len(data_val)>0 & (cnt_per_epoch%val_after_batch==0):
                    del batch
                    del label
                    val_data, val_label = data_handler.return_val()
                    val_data, val_label = val_data.float().to(device), val_label.float().to(device)
                    val_loss = self.validate(val_data, val_label, loss_function=loss_epoch)
                    self.Val_loss.append((cnt, val_loss.item()))
                    if self.metric is not None:
                        oracle_valloss = self.validate(val_data, val_label, loss_function=loss_epoch)
                        oracle_valloss = oracle_valloss.item()
                        self.Oracle_valloss.append((cnt, oracle_valloss))
                    else:
                        oracle_valloss = np.nan
                    del val_data
                    del val_label
                    if val_loss < min_val_loss_epoch:
                        min_val_loss_epoch = val_loss

                self.batch_time.append(timer()-batch_time_start)  ### TIMER

                if np.isnan(loss.item()): # assert(~np.isnan(loss.item())),("Loss goes to NaN!")
                    print(f"Loss goes to NaN! Fail after {cnt} batches.")
                    self.complete = False
                    return

                if (cnt_per_epoch%20==0 or cnt_per_epoch==max_cnt_per_epoch) & (self.vb):
                    _, _, eta = self._training_time(epoch-ep-1, max_cnt_per_epoch-cnt_per_epoch, max_cnt_per_epoch) # TIMER
                    if len(data_val)>0:
                        prt_loss = f'Loss/Val_loss: {round(loss.item(),4)}/{round(val_loss.item(),4)}'
                    else:
                        prt_loss = f'Training loss: {round(loss.item(),4)}'
                    prt_oracle_loss = f'Oracle: {round(oracle_valloss,4)}'
                    prt_num_samples = f'{cnt_per_epoch*batch_size/1000}k/{max_cnt_per_epoch*batch_size/1000}k'
                    prt_num_epoch = f'Epoch {ep+1}/{epoch}'
                    prt_eta = f'ETA {eta}'
                    print('\r'+prt_loss+', '+prt_num_samples+', '+prt_num_epoch+', '+prt_oracle_loss+f', Ktop: {k_top}, '+prt_eta+'     ', end='')

            if min_val_loss_epoch < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = min_val_loss_epoch
            else:
                epochs_no_improve += 1
            if (self.es > 0) & (epochs_no_improve >= self.es):
                print(f'\nEarly stopping after {self.es} epochs with no improvement.')
                break
            
            self.epoch_time.append(timer()-epoch_time_start)  ### TIMER
            self.lr_scheduler.step()

            if self.cp is not None:
                save_path = os.path.join(self.cp, f'model_ckp_{ep}.pt')
                self.save_checkpoint(self.model, self.optimizer, save_path, epoch=ep, loss=loss)

            print() # end while
        self.complete = True
        print('\nTraining Complete!')

    @staticmethod
    def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, save_path: str, epoch: int, loss):
        save_info = {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss}
        torch.save(save_info, save_path)

    @staticmethod
    def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch:int = checkpoint['epoch']
        loss  = checkpoint['loss']
        return model, optimizer, epoch, loss

    def plot_history_loss(self):
        plt.figure()
        plt.plot(np.linspace(1,len(self.Loss),len(self.Loss)), self.Loss, '.', label='loss')
        if len(self.Val_loss):
            plt.plot(np.array(self.Val_loss)[:,0], np.array(self.Val_loss)[:,1], '.', label='val_loss')
        plt.xlabel('#batch')
        plt.ylabel('Loss')
        plt.legend()
