import torch.nn as nn
from models.yolov5_head import Yolov5_Head
from models.yolo_pan import YOLOPAFPN
from utils.torch_utils import to_cpu

class YOLO3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = YOLOPAFPN(depth=config.MODEL.DEPTH , width=config.MODEL.WIDTH)
        self.head     = Yolov5_Head(config)

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)

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