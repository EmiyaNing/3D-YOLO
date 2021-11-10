import torch 
import torch.nn as nn

from models.backbone3d import DarkNet3d
from models.yolo_pan import YOLO3DPAN
from models.yolov5_head import Yolov5_Head

from utils.torch_utils import to_cpu

class SparseYOLO3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = DarkNet3d(config)
        self.neck     = YOLO3DPAN(config.MODEL.DEPTH, config.MODEL.WIDTH)
        self.head     = Yolov5_Head(config)



    def forward(self, features, coors, num_voxel, targets=None):
        BEVResult = self.backbone(features, coors, num_voxel)
        fpn_outs    = self.neck(BEVResult)
        for element in fpn_outs:
            print("In fpn_outs.shape = ", element.shape)
        if self.training:
            assert targets is not None
            loss, outputs = self.head(fpn_outs, targets)
            outputs = to_cpu(outputs)
            return loss, outputs
        else:
            outputs = self.head(fpn_outs)
            outputs = to_cpu(outputs)
            return outputs
    

    def get_metrix(self):
            output = {}
            output["yolo_layer1"] = self.head.end_layer1.metrics
            output["yolo_layer2"] = self.head.end_layer2.metrics
            output["yolo_layer3"] = self.head.end_layer3.metrics
            return output