"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.31
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset
"""

import sys
import os
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2

sys.path.append('../')

from data_process import transformation, kitti_bev_utils, kitti_data_utils
from voxelgenerator import VoxelGenerator
#from spconv.utils import VoxelGenerator
import config.kitti_config as cnf


class KittiVoxelDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', lidar_transforms=None, aug_transforms=None, multiscale=False,
                 num_samples=None, mosaic=False, random_padding=False):
        root_path = os.path.abspath(os.path.dirname('3D-YOLO')).split('3D-YOLO')[0]
        root_path = os.path.join(root_path, '3D-YOLO')
        self.dataset_dir = root_path + dataset_dir
        
        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.lidar_transforms = lidar_transforms
        self.aug_transforms = aug_transforms
        self.batch_count = 0
        self.voxel_generator = VoxelGenerator(
                                                voxel_size = [0.05, 0.05, 0.25],
                                                point_cloud_range = [cnf.boundary_voxel["minX"],
                                                                     cnf.boundary_voxel["minY"],
                                                                     cnf.boundary_voxel["minZ"],
                                                                     cnf.boundary_voxel["maxX"],
                                                                     cnf.boundary_voxel["maxY"],
                                                                     cnf.boundary_voxel["maxZ"]],
                                                max_num_points = 25,
                                                max_voxels=20000
                                            )

        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]

        if self.is_test:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        else:
            self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_lader_only(index)
        else:
            return self.load_lader_with_targets(index)

    def load_lader_only(self, index):
        """Load only image for the testing phase"""

        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        lidarData = kitti_bev_utils.removePoints(lidarData, cnf.boundary_voxel)
        features, coordinates, num_per_voxel = self.voxel_generator.generate(lidarData)
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

        return img_file, features, coordinates, num_per_voxel

    def load_lader_with_targets(self, index):
        """Load images and targets for the training and validation phase"""

        sample_id = int(self.sample_id_list[index])

        lidarData = self.get_lidar(sample_id)
        objects = self.get_label(sample_id)
        calib = self.get_calib(sample_id)

        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)
        # convert rect cam to velo cord
        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P)

        if self.lidar_transforms is not None:
            lidarData, labels[:, 1:] = self.lidar_transforms(lidarData, labels[:, 1:])

        lidarData = kitti_bev_utils.removePoints(lidarData, cnf.boundary_voxel)
        features, coordinates, num_per_voxel = self.voxel_generator.generate(lidarData)
        
        #rgb_map = kitti_bev_utils.makeBVFeature(lidarData, cnf.DISCRETIZATION, cnf.boundary)
        target = kitti_bev_utils.build_yolo_target(labels)
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

        # on image space: targets are formatted as (box_idx, class, x, y, z, h, w, l, im, re)
        n_target = len(target)
        targets = torch.zeros((n_target, 10))
        if n_target > 0:
            targets[:, 1:] = torch.from_numpy(target)

        features = torch.from_numpy(features)
        coordinates = torch.from_numpy(coordinates)
        num_per_voxel = torch.from_numpy(num_per_voxel)


        return img_file, features, coordinates, num_per_voxel, targets


    def __len__(self):
        return len(self.sample_id_list)

    def remove_invalid_idx(self, image_idx_list):
        """Discard samples which don't have current training class objects, which will not be used for training."""

        sample_id_list = []
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)
            if not noObjectLabels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                                   calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_point_cloud_range(labels[i, 1:4]):
                        valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)

        return sample_id_list

    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary_voxel["minX"], cnf.boundary_voxel["maxX"]]
        y_range = [cnf.boundary_voxel["minY"], cnf.boundary_voxel["maxY"]]
        z_range = [cnf.boundary_voxel["minZ"], cnf.boundary_voxel["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def collate_fn(self, batch):
        paths, features, coors, num_voxel, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        # Resize images to input shape
        features = torch.cat(features, 0)
        new_coor = []
        for i, coor in enumerate(coors):
            pad_coor    = torch.full([coor.shape[0], 1], i)
            new_coor.append(torch.cat([pad_coor, coor], 1))
        coors    = torch.cat(new_coor, dim=0)
        num_voxel= torch.cat(num_voxel, 0)
        self.batch_count += 1

        return paths, features, coors, num_voxel, targets

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        # assert os.path.isfile(img_file)
        return cv2.imread(img_file)  # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return kitti_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(label_file)
        return kitti_data_utils.read_label(label_file)


if __name__ == '__main__':
    import argparse
    import os
    import time

    import cv2
    import numpy as np

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    from torch.utils.data import DataLoader
    import config.kitti_config as cnf
    from models.backbone3d import DarkNet3d
    from models.yolo3d import SparseYOLO3D

    import sys
    sys.path.append("..")
    from configs import get_config
    configs = get_config()
    configs.DATA.BATCH_SIZE = 4
    dataset  = KittiVoxelDataset(configs.DATA.DATA_PATH, mode='train', lidar_transforms=None,
                                 aug_transforms=None, multiscale=configs.DATA.MULTISCALE,
                                 num_samples=configs.DATA.NUM_SAMPLE, mosaic=configs.TRAIN.MOSASIC,
                                 random_padding=configs.DATA.RANDOM_PAD)
    dataloader = DataLoader(dataset, batch_size=configs.DATA.BATCH_SIZE, shuffle=True,
                                num_workers=configs.DATA.NUM_WORKERS, sampler=None,
                                collate_fn=dataset.collate_fn)

    #model    = DarkNet3d(configs)
    model     = SparseYOLO3D(configs)
    model.eval()
    for idx, data in enumerate(dataloader):
        img      = data[0]
        features = data[1]
        coors    = data[2]
        num_voxel= data[3]
        target   = data[4]
        start_t  = time.time()
        result   = model(features, coors.int(), num_voxel)
        end_time = time.time()
        print("Now the spent time = ", end_time - start_t)
        '''for key in result.keys():
            print("Now the ", key, ".shape = ",result[key].shape)'''
        for element in result:
            print("Now the output's shape = ", element.shape)

        
        
    