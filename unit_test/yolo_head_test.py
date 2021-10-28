import sys
import argparse
import torch
sys.path.append('..')
from models.yolo_head import YOLOXHead
from models.yolo_fpn import YOLOFPN
from models.yolo_pan import YOLOPAFPN
from models.yolox import YOLOX
from models.yolov5 import YOLO3D
from utils.boxes import postprocess
from configs import get_config
configs = get_config()

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

def test_yolov5_model():
    model = YOLO3D(configs)
    model.eval()
    image = torch.rand([4, 3, 608, 608], dtype=torch.float32)
    output= model(image)
    for element in output:
        print(element.shape)

def test_yolov5_model_train():
    model = YOLO3D(configs)
    model.train()
    image = torch.rand([4, 3, 608, 608], dtype=torch.float32)
    target= torch.rand([15, 9], dtype=torch.float32)
    idx   = torch.ones([15, 1], dtype=torch.float32)
    target= torch.cat([idx, target], -1)
    loss, output = model(image, target)
    print(loss)

if __name__ == '__main__':
    config = parse_configs()
    if config.function == 'head':
        test_yolo_head()
    elif config.function == 'model':
        test_yolox_model()
    elif config.function == 'train':
        test_yolox_head_training()
    elif config.function == 'yolov5':
        test_yolov5_model()
    elif config.function == 'yolov5-train':
        test_yolov5_model_train()
    else:
        print("Not Implement")