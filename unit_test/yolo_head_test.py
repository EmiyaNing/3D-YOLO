import sys
import argparse
import torch
sys.path.append('..')
from models.yolo_head import YOLOXHead
from models.yolo_fpn import YOLOFPN
from models.yolo_pan import YOLOPAFPN
from models.yolox import YOLOX
from utils.boxes import postprocess

def parse_configs():
    parser = argparse.ArgumentParser(description='YOLOX unit_test config')
    parser.add_argument('--function', type=str, default='head',
                        help='The apr used to select test function')
    configs = parser.parse_args()
    return configs

def test_yolo_head():
    model = YOLOPAFPN()
    head  = YOLOXHead(14)
    head.eval()
    model.eval()
    image = torch.rand([4, 3, 604, 604], dtype=torch.float32)
    result = model(image)
    output = head(result)
    print(output)

def test_yolox_model():
    model = YOLOX()
    model.eval()
    image = torch.rand([4, 3, 604, 604], dtype=torch.float32)
    result = model(image)
    output = postprocess(result, 80)
    print(result.shape)

def test_yolox_head_training():
    model = YOLOX()
    model.train()
    image = torch.rand([4, 3, 604, 604], dtype=torch.float32)
    target= torch.randint([4, 10, 5]).type(torch.float32)
    output = model(image, target)
    print('output = ', output)

if __name__ == '__main__':
    config = parse_configs()
    if config.function == 'head':
        test_yolo_head()
    elif config.function == 'model':
        test_yolox_model()
    elif config.function == 'train':
        test_yolox_head_training()
    else:
        print("Not Implement")