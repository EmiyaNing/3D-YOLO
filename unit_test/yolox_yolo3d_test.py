import torch
import sys
sys.path.append("..")
from models.yolox import YOLOX
from models.yolo3dx import YOLO3DX
from configs import get_config
config = get_config()

yolo2d = YOLOX()

yolo3d = YOLO3DX(config)
yolo2d.train()

yolo3d.train()

data1 = torch.zeros([1, 3, 128, 128])
target1 = torch.ones(1, 15, 5)
target2 = torch.ones(1, 15, 8)
result2d = yolo2d(data1, target1)

result3d = yolo3d(data1, target2)
print(result2d)
print(result3d)

