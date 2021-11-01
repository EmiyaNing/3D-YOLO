import spconv
from torch import nn
from models.net_blocks import SiLU

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
        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)


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


class VxNet(nn.Module):
    '''
        Most common used Conv3d backbone in 3d detection.
    '''
    def __init__(self, num_input_features):
        super().__init__()
        self.num_input = num_input_features
        self.conv0 = double_conv(self.num_input, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')              # 2

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')              # 4

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 64, 'down2')              # 8

        self.conv3 = triple_conv(64, 64, 'subm3')

        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (1, 1, 1), (1, 1, 1), bias=False),  # shape no change
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            SiLU()
        )

    def forward(self, inputs):
        middle = []
        x = self.conv0(inputs)
        x = self.down0(x)
        x = self.conv1(x)
        middle.append(x)
        x = self.down1(x)
        x = self.conv2(x)
        middle.append(x)
        x = self.down2(x)
        x = self.conv3(x)
        x = self.extra_conv(x)
        middle.append(x)

        return middle
