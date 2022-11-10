import os, sys
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from network_manager import NetworkManager
from motion_prediction.data_handle import data_handler as dh

from motion_prediction.util import utils_yaml
from motion_prediction.util import utils_np

import pickle
from datetime import datetime

def check_device():
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count(),
              'Current 1:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
    else:
        print(f'CUDA not working! Pytorch: {torch.__version__}.')
        sys.exit(0)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

def load_param(root_dir, config_file, param_in_list=True):
    if param_in_list:
        param_list = utils_yaml.from_yaml_all(os.path.join(root_dir, 'config/', config_file))
        return {**param_list[0], **param_list[1], **param_list[2], **param_list[3]}
    else:
        return utils_yaml.from_yaml(os.path.join(root_dir, 'config/', config_file))

def load_path(param, root_dir):
    save_path = os.path.join(root_dir, param['model_path'])
    csv_path  = os.path.join(root_dir, param['label_path'])
    data_dir  = os.path.join(root_dir, param['data_path'])
    return save_path, csv_path, data_dir

def load_data(param, paths, Dataset, transform, validation_cache=10, load_for_test=False, num_workers=0):
    myDS = Dataset(csv_path=paths[1], root_dir=paths[2], transform=transform, 
                   dynamic_env=param['dynamic_env'], pred_traj=(param['pred_len']>1), T_channel=True)
    myDH = dh.DataHandler(myDS, batch_size=param['batch_size'], validation_cache=validation_cache, num_workers=num_workers)
    if not load_for_test:
        print("Data prepared. #Samples(training, val):{}, #Batches:{}".format(myDH.return_length_ds(), myDH.return_length_dl()))
    print('Sample: {\'image\':',myDS[0]['image'].shape,'\'label\':',myDS[0]['label'],'}')
    return myDS, myDH

def load_manager(param, Net, loss):
    net = Net(param['input_channel'], param['dim_out'], param['fc_input'], num_components=param['num_hypos'])
    myNet = NetworkManager(net, loss, early_stopping=param['early_stopping'], device=param['device'])
    myNet.build_Network()
    return myNet

def save_profile(manager, save_path='./'):
    now = datetime.now()
    dt = now.strftime("%d_%m_%Y__%H_%M_%S")

    manager.plot_history_loss()
    plt.savefig(os.path.join(save_path, dt+'.png'), bbox_inches='tight')
    plt.close()

    loss_dict = {'loss':manager.Loss, 'val_loss':manager.Val_loss}
    with open(os.path.join(save_path, dt+'.pickle'), 'wb') as pf:
        pickle.dump(loss_dict, pf)

def main_train(root_dir, config_file, Dataset, transform, Net, loss, k_top_list, num_workers:int):
    ### Check and load
    check_device()
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir)
    _, myDH = load_data(param, paths, Dataset, transform, num_workers=num_workers)
    myNet = load_manager(param, Net, loss)
    myNet.build_Network()

    ### Training
    start_time = time.time()
    myNet.train(myDH, param['batch_size'], param['epoch'], k_top_list=k_top_list, val_after_batch=param['batch_size'])
    total_time = round((time.time()-start_time)/3600, 4)
    if (paths[0] is not None) & myNet.complete:
        torch.save(myNet.model.state_dict(), paths[0])
    nparams = sum(p.numel() for p in myNet.model.parameters() if p.requires_grad)
    print("\nTraining done: {} parameters. Cost time: {}h.".format(nparams, total_time))

    save_profile(myNet)

def load_net(root_dir, config_file, Net):
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir)
    myNet = load_manager(param, Net, {})
    myNet.build_Network()
    myNet.model.load_state_dict(torch.load(paths[0]))
    myNet.model.eval() # with BN layer, must run eval first
    return myNet

def main_test_pre(root_dir, config_file, Dataset, transform, Net):
    ### Check and load
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir)
    myDS, myDH = load_data(param, paths, Dataset, transform, load_for_test=True)
    if Net is not None:
        myNet = load_manager(param, Net, {})
        myNet.build_Network()
        myNet.model.load_state_dict(torch.load(paths[0]))
        myNet.model.eval() # with BN layer, must run eval first
    else:
        myNet = None
    return myDS, myDH, myNet

def main_test(dataset, net, idx, offset=None):
    img   = dataset[idx]['image']
    label = dataset[idx]['label']
    traj  = dataset[idx]['traj']
    index = dataset[idx]['index']
    if offset is not None:
        img[-1,:,:] = offset*torch.ones_like(img[-1,:,:])
    pred = net.inference(img.unsqueeze(0))
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


