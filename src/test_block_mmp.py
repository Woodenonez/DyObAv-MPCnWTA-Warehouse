import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import torch
import torchvision

from motion_prediction.net_module.net import ConvMultiHypoNet, ConvMixtureDensityNet
from motion_prediction.data_handle import data_handler as dh
from motion_prediction.data_handle import dataset as ds

from motion_prediction.util import utils_test
from motion_prediction.util import zfilter
from motion_prediction.util import utils_np

import motion_prediction.pre_load as pre_load

print("Program: animation\n")

### Config
root_dir = Path(__file__).parents[1]
# config_file = 'bsd_1t20_test.yml'
config_file = 'wsd_1t20_test.yml'

composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = ConvMultiHypoNet
PRED_OFFSET = 20

### Prepare
dataset, _, net = pre_load.main_test_pre(root_dir, config_file, Dataset, composed, Net)

### Visualization option
idx_start = 0
idx_end = len(dataset)
pause_time = 0

### Visualize
fig, ax = plt.subplots()
idc = np.linspace(idx_start, idx_end, num=idx_end-idx_start).astype('int')
for idx in idc:

    # if skip == False:
    #     skip = True
    # else:
    #     skip = False
    #     continue
    print(idx)

    plt.cla()
    label_list = []
    hypos_list = []
    for i in range(1, PRED_OFFSET+1):
        img, label, traj, index, hyposM, reference = pre_load.main_test(dataset, net, idx=idx, offset=i)
        occupancy_map = 255 - reference[:,:,0].float().numpy()
        hyposM = utils_np.get_closest_edge_point(hyposM[0], occupancy_map) # post-processing
        hyposM = hyposM[np.newaxis, :, :]
        label_list.append(label)
        hypos_list.append(hyposM)
    traj = np.array(traj)

    ### Kalman filter
    P0 = zfilter.fill_diag((1,1,1,1))
    Q  = zfilter.fill_diag((0.1,0.1,0.1,0.1))
    R  = zfilter.fill_diag((0.1,0.1))
    KF_X, KF_P        = utils_test.fit_KF(zfilter.model_CV, traj, P0, Q, R, PRED_OFFSET)

    ### CGF
    hypos_clusters_list = []
    mu_list_list  = []
    std_list_list = []
    for i in range(PRED_OFFSET):
        hyposM = hypos_list[i]
        hypos_clusters    = utils_test.fit_DBSCAN(hyposM[0], eps=20, min_sample=3) # DBSCAN
        mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting
        hypos_clusters_list.append(hypos_clusters)
        mu_list_list.append(mu_list)
        std_list_list.append(std_list)

    ### Vis
    ax.imshow(occupancy_map, cmap='Greys')
    utils_test.plot_markers(ax, traj, None, None)
    for i in range(PRED_OFFSET):
        mu_list  = mu_list_list[i]
        std_list = std_list_list[i]
        hyposM = hypos_list[i]
        hypos_clusters = hypos_clusters_list[i]
        utils_test.plot_markers(ax, None, hyposM, None)
        # utils_test.plot_Gaussian_ellipses(ax, mu_list, std_list)
    label = label_list[0][0]
    ax.plot(label[0], label[1], 'mo')
    utils_test.set_axis(ax, title='SWTA')

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    if pause_time==0:
        plt.pause(0.01)
        while not plt.waitforbuttonpress():  # XXX press a button to continue
            pass
    else:
        plt.pause(pause_time)

plt.show()

