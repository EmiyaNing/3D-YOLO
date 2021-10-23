import sys
import torch
sys.path.append("..")
from models.yolo3d_head import YOLOX_3DHead
from models.darknet import CSPDarknet
from models.yolo_pan import YOLOPAFPN

def test_yolox_3dhead():
    data1 = torch.rand([4, 256, 75, 75])
    data2 = torch.rand([4, 512, 37, 37])
    data3 = torch.rand([4, 1024, 19, 19])
    input = (data1, data2, data3)
    model = YOLOX_3DHead(3)
    model.eval()
    res   = model(input)
    print(res.shape)
    print(res)


def test_yolox_3dhead_train():
    backbone = YOLOPAFPN()
    head     = YOLOX_3DHead(3)
    head.train()
    data     = torch.rand([4, 3, 604, 604], dtype=torch.float32)
    target_box   = torch.randint(0, 75, [4, 10, 6]).type(torch.float32)
    target_cls   = torch.randint(0, 3, [4, 10, 1]).type(torch.float32)
    target_yaw   = torch.rand([4, 10, 1], dtype=torch.float32)
    target   = torch.cat([target_cls, target_box, target_yaw], 2)
    feature  = backbone(data)
    result   = head(feature, target)
    print(result)

if __name__ == '__main__':
    #test_yolox_3dhead()
    test_yolox_3dhead_train()
