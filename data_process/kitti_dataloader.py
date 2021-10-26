"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.31
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout


def create_train_dataloader(configs):
    """Create dataloader for training"""

    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.DATA.HFLIP_PROB),
        Cutout(n_holes=configs.DATA.CUTOUT_NHOLES, ratio=configs.DATA.CUTOUT_RATIO, fill_value=configs.DATA.CUTOUT_FILL_VALUE,
               p=configs.DATA.CUTOUT_PROB)
    ], p=1.)

    train_dataset = KittiDataset(configs.DATA.DATA_PATH, mode='train', lidar_transforms=train_lidar_transforms,
                                 aug_transforms=train_aug_transforms, multiscale=configs.DATA.MULTISCALE,
                                 num_samples=configs.DATA.NUM_SAMPLE, mosaic=configs.TRAIN.MOSASIC,
                                 random_padding=configs.DATA.RANDOM_PAD)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.DATA.BATCH_SIZE, shuffle=True,
                                num_workers=configs.DATA.NUM_WORKERS, sampler=None,
                                  collate_fn=train_dataset.collate_fn)

    return train_dataloader


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KittiDataset(configs.DATA.DATA_PATH, mode='val', lidar_transforms=None, aug_transforms=None,
                               multiscale=False, num_samples=configs.DATA.NUM_SAMPLE, mosaic=False, random_padding=False)

    val_dataloader = DataLoader(val_dataset, batch_size=configs.DATA.BATCH_SIZE_EVAL, shuffle=False,
                                num_workers=configs.DATA.NUM_WORKERS, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return val_dataloader


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_dataset = KittiDataset(configs.DATA.DATA_PATH, mode='test', lidar_transforms=None, aug_transforms=None,
                                multiscale=False, num_samples=configs.DATA.NUM_SAMPLE, mosaic=False, random_padding=False)
    test_sampler = None

    test_dataloader = DataLoader(test_dataset, batch_size=configs.DATA.BATCH_SIZE, shuffle=False,
                                num_workers=configs.DATA.NUM_WORKERS, sampler=test_sampler)

    return test_dataloader


if __name__ == '__main__':
    import argparse
    import os

    import cv2
    import numpy as np

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    import config.kitti_config as cnf

    import sys
    sys.path.append("..")
    from configs import get_config
    configs = get_config()
    configs.DATA.BATCH_SIZE = 1


    dataloader = create_train_dataloader(configs)
    print('len train dataloader: {}'.format(len(dataloader)))


    print('\n\nPress n to see the next sample >>> Press Esc to quit...')

    for batch_i, (img_files, imgs, targets) in enumerate(dataloader):
        if not (configs.TRAIN.MOSASIC):
            img_file = img_files[0]
            img_rgb = cv2.imread(img_file)
            calib = kitti_data_utils.Calibration(img_file.replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = invert_target(targets[:, 1:], calib, img_rgb.shape, RGB_Map=None)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        # target has (b, cl, x, y, z, h, w, l, im, re)
        targets[:, 2:8] *= configs.DATA.IMG_SIZE
        img_bev = imgs.squeeze() * 255
        print(img_bev.shape)
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.DATA.IMG_SIZE, configs.DATA.IMG_SIZE))

        # Draw rotated box
        for c, x, y, z, h, w, l, im, re in targets[:, 1:].numpy():
            yaw = np.arctan2(im, re)
            bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])

        img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)
        cv2.imshow('img_bev', img_bev)

        '''if configs.TRAIN.MOSASIC:
            cv2.imshow('mosaic_sample', img_bev)
        else:
            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=configs.output_width)
            cv2.imshow('single_sample', out_img)'''
        if cv2.waitKey(10000) & 0xFF == ord('q'):
            break
