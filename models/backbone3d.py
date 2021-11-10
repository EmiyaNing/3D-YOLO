import time
import spconv
import torch
from torch import nn
from models.net_blocks import SiLU,BaseConv

class SimpleVoxel(nn.Module):
    def __init__(self, num_input_features=4, use_norm=True):
        super().__init__()
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels):
        '''
        Args:
            features:   [concated_num_points, num_voxel_size, 4]
            num_voxels: [concated_num_points]
        Return:
            features
        '''
        points_mean = torch.sum(features[:, :, :self.num_input_features], dim=1, keepdim=False)/ num_voxels.type_as(features).view(-1, 1)
        return points_mean


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            SiLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            SiLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            SiLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            SiLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            SiLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            SiLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 3, (2, 2, 2), padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            SiLU()
    )


class BaseConv3d(nn.Module):
    '''
        Basic module of SpareConv3d
    '''
    def __init__(self, in_channels, out_channels, kernal_size = 3, stride=(1, 1, 1), padding=1, indice_key=None):
        super().__init__()
        self.conv = spconv.SparseConv3d(in_channels,
                                        out_channels,
                                        kernal_size,
                                        stride,
                                        padding=padding,
                                        bias=False, 
                                        ndice_key=indice_key)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.act  = SiLU()
    
    def forward(self, inputs):
        return self.act(self.norm(self.conv(inputs)))


class Res3dBottleneck(nn.Module):
    '''
        Basic of res3d block
    '''
    def __init__(self, in_channels, indice_key=None):
        super().__init__()
        hidden_dim = in_channels // 2
        self.conv1 = BaseConv3d(in_channels, hidden_dim, indice_key=indice_key)
        self.conv2 = BaseConv3d(hidden_dim, in_channels, indice_key=indice_key)


    def forward(self, inputs):
        '''
            Basic short connect.
        '''
        return inputs + self.conv2(self.conv1(inputs))


class Res3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num = 1, indice_key=None):
        super().__init__()
        self.stride_conv = stride_conv(in_channels, out_channels, indice_key)
        layer_list = [Res3dBottleneck(out_channels, indice_key) for i in range(layer_num)]
        self.conv  = spconv.SparseSequential(*layer_list)

    def forward(self, inputs):
        result = self.conv(inputs)
        result = self.stride_conv(result)
        return result


class VxNet(nn.Module):
    '''
        Most common used Conv3d backbone in 3d detection.
    '''
    def __init__(self, num_input_features, input_shape):
        super().__init__()
        self.num_input = num_input_features
        self.sparse_shape = input_shape
        self.conv0 = double_conv(self.num_input, 16, 'subm0')
        self.down0 = stride_conv(16, 16, 'down0')              # 2

        self.conv1 = double_conv(16, 16, 'subm1')
        self.down1 = stride_conv(16, 32, 'down1')              # 4

        self.conv2 = triple_conv(32, 64, 'subm2')
        self.down2 = stride_conv(64, 64, 'down2')              # 8

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.to_dense = spconv.ToDense()

        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(128, 256, (1, 1, 1), (1, 1, 1), bias=False),  # shape no change
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.01),
            SiLU()
        )

    def forward(self, voxel_features, coors, batch_size):
        middle = []
        inputs = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x = self.conv0(inputs)
        x = self.down0(x)
        x = self.conv1(x)
        middle.append(self.to_dense(x))
        x = self.down1(x)
        x = self.conv2(x)
        middle.append(self.to_dense(x))
        x = self.down2(x)
        x = self.conv3(x)
        x = self.extra_conv(x)
        middle.append(self.to_dense(x))

        return middle

class ResVxNet(nn.Module):
    def __init__(self, num_input_features, input_shape):
        super().__init__()
        self.num_input = num_input_features
        self.sparse_shape = input_shape
        self.block0 = Res3dBlock(num_input_features, 32, 2)
        self.block1 = Res3dBlock(32, 32, 4)
        self.block2 = Res3dBlock(32, 64, 4)
        self.block3 = Res3dBlock(64, 64, 2)
        self.to_dense = spconv.ToDense()


    def forward(self, voxel_features, coors, batch_size):
        middle = []
        inputs = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x = self.block0(inputs)
        x = self.block1(x)
        middle.append(self.to_dense(x))
        x = self.block2(x)
        middle.append(self.to_dense(x))
        x = self.block3(x)
        middle.append(self.to_dense(x))

        return middle


class DarkNet3d(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.pre_voxel = SimpleVoxel(config.MODEL.INPUT_DIM)
        self.backbone  = VxNet(config.MODEL.INPUT_DIM, config.MODEL.DENSE_DIM)
        self.depth     = config.MODEL.DEPTH
        self.width     = config.MODEL.WIDTH
        self.batch_size= config.DATA.BATCH_SIZE
        # down sample 8 times for feature 1, down size 
        self.down_conv1= nn.Sequential(
            BaseConv(128, 64, 3, 2),
            BaseConv(64, 64, 3, 2),
            BaseConv(64, 128, 3, 2),
            BaseConv(128, int(256 * self.width), 1, 1)
        )
        # down sample 8 times for feature 2
        self.down_conv2= nn.Sequential(
            BaseConv(256, 128, 3, 2),
            BaseConv(128, 128, 3, 2),
            BaseConv(128, 256, 3, 2),
            BaseConv(256, int(512 * self.width), 1, 1)
        )
        # down sample 8 times for feature 3
        self.down_conv3= nn.Sequential(
            BaseConv(512, 256, 3, 2),
            BaseConv(256, 256, 3, 2),
            BaseConv(256, 512, 3, 2),
            BaseConv(512, int(1024 * self.width), 1, 1)
        )

    def forward(self, voxels, coors, num_voxels):
        output   = {}
        voxels   = self.pre_voxel(voxels, num_voxels)
        features = self.backbone(voxels, coors, self.batch_size)
        out1     = self.down_conv1(features[0].view(features[0].shape[0], 
                                                    features[0].shape[1] * features[0].shape[2],
                                                    features[0].shape[3],
                                                    features[0].shape[4]))
        output["dark3"] = out1
        out2     = self.down_conv2(features[1].view(features[1].shape[0],
                                                    features[1].shape[1] * features[1].shape[2],
                                                    features[1].shape[3],
                                                    features[1].shape[4]))
        output["dark4"] = out2
        out3     = self.down_conv3(features[2].view(features[2].shape[0],
                                                    features[2].shape[1] * features[2].shape[2],
                                                    features[2].shape[3],
                                                    features[2].shape[4]))
        output["dark5"] = out3
        return output
