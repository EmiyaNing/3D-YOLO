import sys
import argparse
import torch
sys.path.append('..')
from models.yolo_fpn import YOLOFPN
from models.yolo_pan import YOLOPAFPN

def parse_configs():
    parser = argparse.ArgumentParser(description='YOLOX unit_test config')
    parser.add_argument('--function', type=str, default='fpn',
                        help='The apr used to select test function')
    configs = parser.parse_args()
    return configs

def test_yolo_fpn():
    model = YOLOFPN()
    image = torch.rand([4, 3, 604, 604], dtype=torch.float32)
    result = model(image)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)

def test_yolo_pafpn():
    model = YOLOPAFPN()
    image = torch.rand([4, 3,604, 604], dtype=torch.float32)
    result = model(image)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)

if __name__ == '__main__':
    config = parse_configs()
    if config.function == 'fpn':
        test_yolo_fpn()
    elif config.function == 'pan':
        test_yolo_pafpn()
    else:
        print("Not implement.....")