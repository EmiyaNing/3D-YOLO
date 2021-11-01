import math
import torch
import torch.nn as nn
from models.yolo_layer import YoloLayer
from configs import get_config

class Yolov5_Head(nn.Module):
    def __init__(self, configs, width=1.0):
        super().__init__()
        self.num_anchors = configs.MODEL.NUM_ANCHR 
        self.num_classes = configs.MODEL.NUM_CLASSES
        self.img_size    = configs.DATA.IMG_SIZE
        self.anchors_layer1 = [
            (259, 10, 10, 0, 1),
            (269, 11, 13, 0, 1),
            (275, 13, 16, 0, 1)
        ]
        self.anchors_layer2 = [
            (261, 8, 24, 0, 1),
            (261, 11, 24, 0, 1),
            (273, 11, 26, 0, 1)
        ]
        self.anchors_layer3 = [
            (221, 23, 49, 0, 1),
            (249, 23, 52, 0, 1),
            (323, 26, 64, 0, 1)
        ]
        self.stem_conv1  = nn.Sequential(
            nn.Conv2d(in_channels = int(256 * width),
                      out_channels = self.num_anchors * (8 + 1 + self.num_classes),
                      kernel_size = 1,
                      stride = 1,
                      padding = 0),
        )
        self.stem_conv2  = nn.Sequential(
            nn.Conv2d(in_channels = int(512 * width),
                      out_channels = self.num_anchors * (8 + 1 + self.num_classes),
                      kernel_size = 1,
                      stride = 1,
                      padding = 0),
        )
        self.stem_conv3  = nn.Sequential(
            nn.Conv2d(in_channels = int(1024 * width),
                      out_channels = self.num_anchors * (8 + 1 + self.num_classes),
                      kernel_size = 1,
                      stride = 1,
                      padding = 0),
        )
        self.end_layer1  = YoloLayer(
            self.num_classes, self.anchors_layer1, 8, 1.2, ignore_thresh=0.7
        )
        self.end_layer2  = YoloLayer(
            self.num_classes, self.anchors_layer2, 16, 1.1, ignore_thresh=0.7
        )
        self.end_layer3  = YoloLayer(
            self.num_classes, self.anchors_layer3, 32, 1.05, ignore_thresh=0.7
        )


    def forward(self, inputs, targets=None):
        stride8  = inputs[0]
        stride16 = inputs[1]
        stride32 = inputs[2]
        temp_8   = self.stem_conv1(stride8)
        temp_16  = self.stem_conv2(stride16)
        temp_32  = self.stem_conv3(stride32)
        out_stride_8, loss1  = self.end_layer1(temp_8, targets, self.img_size, False)
        out_stride_16, loss2 = self.end_layer2(temp_16, targets, self.img_size, False)
        out_stride_32, loss3 = self.end_layer3(temp_32, targets, self.img_size, False)
        total_loss = loss1 + loss2 + loss3
        output = torch.cat([out_stride_8, out_stride_16, out_stride_32], 1)
        if self.training:
            return total_loss, output
        else:
            return output



if __name__ == '__main__':
    configs = get_config()
    model = Yolov5_Head(configs)
    print(model)