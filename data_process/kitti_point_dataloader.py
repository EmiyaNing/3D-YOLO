import sys
from torch.utils.data import DataLoader

sys.path.append('../')
from data_process.kitti_point_dataset import KittiVoxelDataset
from data_process.transformation import  Random_Rotation, Random_Scaling, OneOf

def create_point_train_loader(configs):

    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=10., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    dataset  = KittiVoxelDataset(configs.DATA.DATA_PATH, mode='train', lidar_transforms=None,
                                 aug_transforms=None, multiscale=configs.DATA.MULTISCALE,
                                 num_samples=configs.DATA.NUM_SAMPLE, mosaic=configs.TRAIN.MOSASIC,
                                 random_padding=configs.DATA.RANDOM_PAD)

    dataloader = DataLoader(dataset, batch_size=configs.DATA.BATCH_SIZE, shuffle=True,
                                num_workers=configs.DATA.NUM_WORKERS, sampler=None,
                                collate_fn=dataset.collate_fn)
    return dataloader


def create_point_test_loader(configs):
    dataset  = KittiVoxelDataset(configs.DATA.DATA_PATH, mode='test', lidar_transforms=None,
                                 aug_transforms=None, multiscale=configs.DATA.MULTISCALE,
                                 num_samples=configs.DATA.NUM_SAMPLE, mosaic=configs.TRAIN.MOSASIC,
                                 random_padding=configs.DATA.RANDOM_PAD)
                                 
    dataloader = DataLoader(dataset, batch_size=configs.DATA.BATCH_SIZE, shuffle=True,
                                num_workers=configs.DATA.NUM_WORKERS, sampler=None,
                                collate_fn=dataset.collate_fn)
    return dataloader


def create_point_val_loader(configs):
    dataset  = KittiVoxelDataset(configs.DATA.DATA_PATH, mode='val', lidar_transforms=None,
                                 aug_transforms=None, multiscale=configs.DATA.MULTISCALE,
                                 num_samples=configs.DATA.NUM_SAMPLE, mosaic=configs.TRAIN.MOSASIC,
                                 random_padding=configs.DATA.RANDOM_PAD)
                                 
    dataloader = DataLoader(dataset, batch_size=configs.DATA.BATCH_SIZE, shuffle=True,
                                num_workers=configs.DATA.NUM_WORKERS, sampler=None,
                                collate_fn=dataset.collate_fn)
    return dataloader



if __name__ == '__main__':
    import argparse
    import os
    import time

    import cv2
    import numpy as np

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    import config.kitti_config as cnf
    from models.backbone3d import DarkNet3d
    from models.yolo3d import SparseYOLO3D

    import sys
    sys.path.append("..")
    from configs import get_config
    configs = get_config()
    configs.DATA.BATCH_SIZE = 4
    dataloader = create_point_train_loader(configs)

    model     = SparseYOLO3D(configs)
    model.train()
    for idx, data in enumerate(dataloader):
        img      = data[0]
        features = data[1]
        coors    = data[2]
        num_voxel= data[3]
        target   = data[4]
        start_t  = time.time()
        try:
            loss,result   = model(features, coors.int(), num_voxel, target)
        except:
            print(img)
        end_time = time.time()
        print("Now in ", idx, " / ", len(dataloader))
        for element in result:
            print("Now the output's shape = ", element.shape)