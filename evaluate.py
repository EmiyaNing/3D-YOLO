"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.31
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for evaluation

"""

import argparse
import os
import time
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data.distributed
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.append('./')

from data_process.kitti_dataloader import create_val_dataloader
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import post_processing, get_batch_statistics_rotated_bbox, ap_per_class, load_classes, post_processing_v2


def evaluate_mAP(val_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, z, h, w, l, im, re))
            targets[:, 2:4] *= configs.DATA.IMG_SIZE
            targets[:, 5:8] *= configs.DATA.IMG_SIZE
            imgs = imgs.to('cuda', non_blocking=True)
            #print(configs)

            outputs = model(imgs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.EVAL.CONF_THRESH, nms_thresh=configs.EVAL.NMS_THRESH)

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.EVAL.IOU_THRESH )

            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)



            start_time = time.time()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

