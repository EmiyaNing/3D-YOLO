import os
import time
import sys
import numpy as np
import spconv
import torch
sys.path.append('..')
from models.backbone3d import VxNet,SimpleVoxel
from spconv.test_utils import generate_sparse_data

def test_VxNet():
    start_t = time.time()
    data = generate_sparse_data([60, 1600, 1408], [60 * 1600 * 1408 // 2000],
                                    3,
                                    data_range=[-1, 1],
                                    with_dense=False)
    time_1  = time.time()
    features = np.ascontiguousarray(data["features"]).astype(np.float32)
    indices = np.ascontiguousarray(
        data["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
    features = torch.from_numpy(features)
    indices  = torch.from_numpy(indices)
    model2 = VxNet(4, [16, 64, 64])
    #data1  = model1(voxel_data)
    result = model2(features, indices, 1)
    time_2  = time.time()
    for element in result:
        b, c, z, y, x = element.shape
        element = element.view(b, c * z, y, x)
        print(element.shape)
    #spare_data = spconv.SparseConvTensor(voxels)
    print("data generate time = ", time_1 - start_t)
    print("model forward consume time = ", time_2 - time_1)



if __name__ == '__main__':
    test_VxNet()