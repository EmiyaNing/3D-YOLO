import argparse
import sys
import time

import cv2
import torch
sys.path.append('..')
from models.yolox import YOLOX
from utils.boxes import postprocess,vis,ValTransform
from utils.coco_classes import COCO_CLASSES


preprocess = ValTransform(legacy=False)

def parse_configs():
    parser = argparse.ArgumentParser(description='YOLOX model test file')
    parser.add_argument('--img_path', type=str, default=None,
                        help='The image file path')
    parser.add_argument('--conf_thre', type=float, default=0.7,
                        help='confidience threshould of model result')
    parser.add_argument('--nms_thre', type=float, default=0.45,
                        help='non max express threshould')
    parser.add_argument('--cls_conf', type=float, default=0.35,
                        help='class confidience of the model')
    parser.add_argument('--num_classes', type=float, default=80,
                        help='model output class number')
    parser.add_argument('--pretrain', type=str, default='../yolox')
    configs = parser.parse_args()
    return configs

def predict(img, config):
    model          = YOLOX()
    model.eval()
    model.load_state_dict(torch.load(config.pretrain, map_location='cpu')['model'])
    number_classes = config.num_classes
    conf_thre      = config.conf_thre
    nms_thre       = config.nms_thre

    # ether the img is a filename or image file
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = img
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    ratio = min(604 / img.shape[0], 604 / img.shape[1])
    img_info["ratio"] = ratio

    img,_          = preprocess(img, None, [604, 604])
    img            = torch.from_numpy(img).unsqueeze(0)
    img            = img.float()

    with torch.no_grad():
        t0 = time.time()
        outputs = model(img)
        t1 = time.time()
        outputs = postprocess(outputs, number_classes, conf_thre, nms_thre)
        t2 = time.time()
        print("Infer time : {:.4f}s".format(t1 - t0))
        print('Post_Process time : {:.4f}s'.format(t2 - t1))

    return outputs,img_info

def visualize(output, img_info, cls_conf=0.35):
    ratio = img_info['ratio']
    img   = img_info['raw_img']
    if output is None:
        return img
    #output = output.cpu()
    bboxes = output[:, 0:4]
    bboxes /= ratio

    classes = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(img, bboxes, scores, classes, cls_conf, COCO_CLASSES)
    return vis_res

def main():
    config = parse_configs()
    output, img_info = predict(config.img_path, config)
    vis_res = visualize(output[0], img_info, config.cls_conf)
    cv2.imshow('visual_result', vis_res)
    cv2.waitKey()

if __name__ == '__main__':
    main()