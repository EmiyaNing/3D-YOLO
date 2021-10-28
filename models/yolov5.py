import torch.nn as nn
from models.yolov5_head import Yolov5_Head
from models.yolo_pan import YOLOPAFPN

class YOLO3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = YOLOPAFPN()
        self.head     = Yolov5_Head(config)

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, outputs = self.head(fpn_outs, targets)
            return loss, outputs
        else:
            outputs = self.head(fpn_outs)
            return outputs