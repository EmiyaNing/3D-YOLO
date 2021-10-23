'''
This script implement the yolo3d-yolox head.
The yolo3d-yolox head should predict:
    1. object class
    2. whether the object have been truncated(whether the object is out the camera views)
    3. whether the object have been occluded
    4. oritentation of object
    5. 3d box property
    6. 3d box center
    7. rotation_y
    8. object score
'''
# 1. finish this head code
# 2. remove the 2d predict
import math
import sys
sys.path.append('..')
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.boxes import bboxes_iou

from .losses import IOUloss,IOU3Dloss
from .net_blocks import BaseConv, DWConv


class YOLOX_3DHead(nn.Module):
    '''
    Describe:
        This is yolox_3dhead class.
        Use this module to generate the yolox's output.
        In the train mode, this class will output the loss vale.
        In the eval  mode, this class will output decode output value.
        Be careful, the 3d pointcloud's coordinate axis is different with BEV feature's x and y
    '''
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs   = nn.ModuleList() # class branch
        self.reg_convs   = nn.ModuleList() # regresession branch
        self.occlu_convs = nn.ModuleList() # occluded and trunch branch
        self.cls_preds   = nn.ModuleList() # class head in class branch
        self.obj_preds   = nn.ModuleList() # score head in regression branch
        self.reg_preds   = nn.ModuleList() # box3d head x,y,z,h,w,l in regression branch
        self.yaw_preds   = nn.ModuleList() # corner head in regression branch
        self.occlu_preds = nn.ModuleList() # occlu head in occluded and trunch branch
        self.truck_preds = nn.ModuleList() # truck head in occluded and trunch branch
        self.stems       = nn.ModuleList()

        self.use_l1 = False
        self.l1_loss  = nn.L1Loss(reduction="none")
        self.yaw_loss = nn.SmoothL1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOU3Dloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)


        Conv = DWConv if depthwise else BaseConv
        for i in range(len(in_channels)):
            # head start conv
            self.stems.append(BaseConv(int(in_channels[i] * width), int(256 * width), 1, 1, act=act))
            # cls branch
            self.cls_convs.append(
                nn.Sequential(
                    *[Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                      Conv(int(256 * width), int(256 * width), 3, 1, act=act)]
                )
            )
            # reg branch
            self.reg_convs.append(
                nn.Sequential(
                    *[Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                      Conv(int(256 * width), int(256 * width), 3, 1, act=act)]
                )
            )
            # occlu branch
            self.occlu_convs.append(
                nn.Sequential(
                    *[Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                      Conv(int(256 * width), int(256 * width), 3, 1, act=act)]
                )
            )
            # class head
            self.cls_preds.append(
                nn.Conv2d(int(256 * width), self.num_classes, 1, 1, 0)
            )
            self.reg_preds.append(
                nn.Conv2d(int(256 * width), 6, 1, 1, 0)
            )
            self.obj_preds.append(
                nn.Conv2d(int(256 * width), 1, 1, 1, 0)
            )
            self.yaw_preds.append(
                nn.Conv2d(int(256 * width), 1, 1, 1, 0)
            )
            self.occlu_preds.append(
                nn.Conv2d(int(256 * width), 1, 1, 1, 0)
            )
            self.truck_preds.append(
                nn.Conv2d(int(256 * width), 1, 1, 1, 0)
            )



    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        '''
        Describe:
            The top level yolox head forward function.
        Args:
            xin                 backbone feature tensor list[dark3, dark4, dark5]
            
            imgs                original input image
        Return:
            
        '''
        outputs      = []
        x_shifts     = []
        y_shifts     = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, occlu_conv, stride_t, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.occlu_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            occ_x = x
            cls_feat   = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat   = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            yaw_output = self.yaw_preds[k](reg_feat)

            occ_feat   = occlu_conv(occ_x)
            occ_output = self.occlu_preds[k](occ_feat)
            truc_output= self.truck_preds[k](occ_feat)
            if self.training:
                output     = torch.cat([truc_output, occ_output, yaw_output,
                                        reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_t, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1]).fill_(stride_t).type_as(xin[0])
                )
            else:
                output = torch.cat(
                    [truc_output.sigmoid(), occ_output.sigmoid(), yaw_output.sigmoid(),
                     reg_output, obj_output, cls_output.sigmoid()], 1
                )
            outputs.append(output)
        if self.training:
            # get loss
            return self.get_losses(
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                dtype=xin[0].dtype
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            return self.decode_outputs(outputs, dtype=xin[0].type())


    def get_output_and_grid(self, output, k, stride, dtype):
        '''
        This function use to generate grid for BEV.
        Args:
            output          A tensor with shape [batch_size, 10 + num_class, gridx, gridy]
            k               A int value range from 0-3 or 0-4
            stride          A int value that show how many time the BEV narrowed
            dtype           A string that show the output's data type
        Return:
            output          Input tensor add the grid
            grid            A grid
        '''
        grid         = self.grids[k]
        batch_size   = output.shape[0]
        n_ch         = 10 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv   = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid     = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid
        output       = output.permute(0, 2, 3, 1).reshape(
            batch_size, hsize * wsize, -1
        )
        # model's output x represent the BEV's height
        # output y represent the BEV's width
        # model's output w represent the BEV's width axis length
        # model's output l represent the BEV's height axis length
        grid         = grid.view(1, -1, 2)
        output[..., 3:5] = (output[..., 3:5] + grid) * stride
        output[..., 7:9] = torch.exp(output[..., 7:9]) * stride
        return output, grid



    def decode_outputs(self, outputs, dtype):
        '''
        This function use in inference to generate a output stride.
        Each element in yolox's output, add the grid index then multiple the stride.
        Args:
            outputs           A tensor with shape [batchsz, class_num + 5, gridx, gridy]
            dtype
        Returns:
            outputs           A tensor with shape [batchsz, gridx * gridy, class_num+5]
        '''
        grids    = []
        strides  = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids   = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # model's output x represent the BEV's height
        # output y represent the BEV's width
        # model's output w represent the BEV's width axis length
        # model's output l represent the BEV's height axis length
        outputs[..., 3:5] = (outputs[..., 3:5] + grids) * strides
        outputs[..., 7:9] = torch.exp(outputs[..., 7:9]) * strides
        return outputs

    def get_losses(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        dtype):
        # may occur bug
        bbox_preds = torch.cat([outputs[:, :, 3:9], outputs[:, :, 2].unsqueeze(-1)], 2)
        obj_preds  = outputs[:, :, 9].unsqueeze(-1)
        cls_preds  = outputs[:, :, 10:]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 6))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:8]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)


        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        # the loss caculate way should be change.....
        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 7)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        loss_yaw = (
            self.yaw_loss(
                bbox_preds[:, :, -1].view(-1, 1)[fg_masks], reg_targets[:, -1]
            )
        ).sum() / num_fg


        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_yaw

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            num_fg / max(num_gts, 1),
        )


    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        mode="gpu",
    ):
        '''
        This function used to match each ground truth bbox and pred output.
        Then compute the iou_loss and cls loss as cost.
        Finally use the dynamic_k_match, to match the ground truth bbox and pred output.
        Args:
            batch_idx               A int value
            num_gt                  A int value, used to show how many ground truth bbox.
            total_num_anchors       A int value, used to show how many bbox in pred output.
            gt_bboxes_per_image     A float Tensor, with shape [num_gt, 7].
                                    each element was consist of y,x,z,h,w,l,yaw
            gt_class                A float Tensor, which store the ground truth classes.
            bboxes_preds_per_image  A float Tensor, which store the pred bboxes.
                                    each element was consist of x,y,z,h,w,l,yaw
            expanded_strides        A int Tensor, used to show how many time the pred should be enlarge.
            x_shifts                A int Tensor
            y_shifts                A int Tensor
            cls_preds               A float Tensor, which store the pred classes.
            bbox_preds              A useless variable.
            obj_preds               A float Tensor, which store the confidience of each pred bboxes.
        Returns:
            num_fg                  A int value, show how many pred bboxes have matched the groud truth.
            gt_matched_classes      A float Tensor, show the matched bbox's classes.
            pred_ious_this_matching A float Tensor, show the ious of this matched bbox.
            matched_gt_indes        A int Tensor, show the groud truth ids of this bbox.
        '''
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        # get groudtruth bev box
        gt_bev_per_image_xy = gt_bboxes_per_image[:, :2]
        gt_bev_per_image_wl = gt_bboxes_per_image[:, 4:6]
        gt_bev_per_image    = torch.cat([gt_bev_per_image_xy, gt_bev_per_image_wl], 1)
        # get pred bev box
        # fucking the axis translation
        pred_bev_per_image_xy = bboxes_preds_per_image[:, :2]
        pred_bev_per_image_wl = bboxes_preds_per_image[:, 4:6]
        pred_bev_per_iamge  = torch.cat([pred_bev_per_image_xy, pred_bev_per_image_wl], 1)


        # compute bev pair_wise_ious_loss
        pair_wise_ious = bboxes_iou(gt_bev_per_image, pred_bev_per_iamge)
        # get the onehot gt_classes
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )


        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        '''
        This function will generate two mask.
        One mask show whether the grid cell's center was setting in a ground truth bbox.
        Another mask show whether the grid cell's center close to a gruond truth bbox's center.
        Args:
            gt_bboxes_per_image       Yolox head's output, which shape should
                                      be [totla_num_anchors, 7]
                                    
            expanded_strides          A Tensor used to indicate how many time
                                      the result should be enlarged.It's shape
                                      should be [totla_num_anchors]
            x_shifts                  A Tensor with shape [totla_num_anchors], it used
                                      as yolox head's x shift index.
            y_shifts                  A Tensor with shape [totla_num_anchors], it used
                                      as yolox head's y shift index.
            totla_num_anchors         A int value
            num_gt                    A int value, used to show how many ground truth bbox
                                      in this image.
        Returns:
            is_in_boxes_all           A bool Tensor
            is_in_boxes_and_center    A bool Tensor
        '''
        # model's output x represent the BEV's height
        # output y represent the BEV's width
        # model's output w represent the BEV's width axis length
        # model's output l represent the BEV's height axis length
        expanded_strides_per_image = expanded_strides[0]
        bevw_shifts_per_image      = x_shifts[0] * expanded_strides_per_image
        bevh_shifts_per_image      = y_shifts[0] * expanded_strides_per_image
        w_center_per_image         = (
            (bevw_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0).repeat(num_gt, 1)
        )
        h_center_per_image         = (
            (bevh_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0).repeat(num_gt, 1)
        )

        # groud truth's x represent BEV's height
        # groud truth's y represent BEV's width
        # build_yolo_target function's return is y, x, z, h, w ,l, im, re
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 4])
            .unsqueeze(1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 4])
            .unsqueeze(1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 5])
            .unsqueeze(1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 5])
            .unsqueeze(1).repeat(1, total_num_anchors)
        )

        b_l = w_center_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - w_center_per_image
        b_t = h_center_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - h_center_per_image
        bbox_deltas = torch.stack([b_l, b_r, b_t, b_b], 2)
        # whether the grid center is in groudtruth bbox mask
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = w_center_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - w_center_per_image
        c_t = h_center_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - h_center_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center


    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds