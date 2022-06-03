import numpy as np
import cv2
import os
from dataset_kitti_odometry import KITTI_Odometry

scene = "00"
evalset = KITTI_Odometry(scene=scene)
scene_info=evalset._build_dataset()

data_info=scene_info['00'];
poses=data_info['poses_left'];
#print(poses[0])
#print(data_info['n_frames'])
selectors={};
for num in range(10):
    selector_name = 'kitti-sub'+format(str(num), '0>2s')
    selectors[selector_name]={
        "sample_stride": 1,
        "train": lambda length,cell_num=num: [i for i in range(cell_num*min(100, length), (cell_num+1)*min(100, length), 1) if i % 10],
        "val": lambda length,cell_num=num: [i for i in range(cell_num*min(100, length), (cell_num+1)*min(100, length), 10)],
        "test": lambda length,cell_num=num: [i for i in range(cell_num*min(100, length), (cell_num+1)*min(100, length), 1)],
    }
    
#print(selectors['kitti-sub00']['train'](2000))
#print(selectors['kitti-sub00']['val'](2000))