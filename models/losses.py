import sys
import torch
import torch.nn as nn
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

sys.path.append("..")
from utils.cal_intersection_rotated_boxes import intersection_area, PolyArea2D
from utils.iou_rotated_boxes_utils import cvt_box_2_polygon, get_corners_3d


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class IOU3Dloss(nn.Module):
    """
        Only caculate the BEV iou loss.
    """
    def __init__(self, reduction="none", loss_type="iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 8)
        target = target.view(-1, 8)
        pred = torch.cat([pred[:, 0:2], pred[:, 4:6]], 1)
        target = torch.cat([target[:, 0:2], target[:, 4:6]], 1)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.SmoothL1Loss()

    def forward(self, pred, target):
        n_boxes = pred.size(0)
        t_x, t_y, t_z, t_h, t_w, t_l, t_yaw = target.t()
        p_x, p_y, p_z, p_h, p_w, p_l, p_yaw = pred.t()
        loss_x = self.crit(p_x, t_x).mean()
        loss_y = self.crit(p_y, t_y).mean()
        loss_z = self.crit(p_z, t_z).mean()
        loss_h = self.crit(p_h, t_h).mean()
        loss_w = self.crit(p_w, t_w).mean()
        loss_l = self.crit(p_l, t_l).mean()
        loss_yaw = self.crit(p_yaw, t_yaw).mean()
        loss_all = loss_x + loss_y + loss_z + loss_h + loss_w + loss_l + loss_yaw
        return loss_all