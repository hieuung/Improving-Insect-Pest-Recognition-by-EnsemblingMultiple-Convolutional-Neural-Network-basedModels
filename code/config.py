"""This code is refer to https://github.com/ZF4444/MMAL-Net"""

from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
import numpy as np
import os


CUDA_VISIBLE_DEVICES = '0'  # The current version only supports one GPU training

model_name = 'Fine-grained'

batch_size = 6
vis_num = batch_size  # The number of visualized images in tensorboard
eval_trainset = False  # Whether or not evaluate trainset
max_checkpoint_num = 150
end_epoch = 150
init_lr = 0.001
lr_milestones = [60, 100]
lr_decay_rate = 0.1
weight_decay = 1e-4
stride = 32
channels = 2048
input_size = 448

N_list = [3, 2, 1]
proposalN = sum(N_list)  # proposal window num
window_side = [192, 256, 320]
iou_threshs = [0.25, 0.25, 0.25]
ratios = [[6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
              [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]

model_path = 'insect_recognition'      # pth save path
weight_path = os.path.join(os.getcwd(), 'pre-trained')

'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]

window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]
