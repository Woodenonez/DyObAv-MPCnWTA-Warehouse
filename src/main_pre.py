import os
from pathlib import Path

from motion_prediction.data_handle.dataset import io, ImageStackDataset

from util import utils_sl
from util.basic_datatype import *

def prepare_map(scene:str, root_dir:FolderDir, inversed_pixel:bool=False):
    scene_dir_name = scene.lower()+'_sim_original'
    map_path     = os.path.join(root_dir, 'data', scene_dir_name, 'mymap.pgm')
    ref_map_path = os.path.join(root_dir, 'data', scene_dir_name, 'label.png')
    with open (map_path, 'rb') as pgmf:
        the_map = utils_sl.read_pgm_and_process(pgmf, inversed_pixel=inversed_pixel)
    ref_map = ImageStackDataset.togray(io.imread(ref_map_path))

    if scene.lower() is 'bookstore':
        the_map[0:40, 950:980] = 0 # for the bookstore scene
    return the_map, ref_map

def prepare_params(config_file_name:str, root_dir:FolderDir):
    cfg_path = os.path.join(root_dir, 'config', config_file_name)
    param_dict = utils_sl.from_yaml(cfg_path)
    return param_dict
