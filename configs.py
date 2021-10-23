import os
from yacs.config import CfgNode as CN
import yaml


_C = CN()
_C.BASE = ['']

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 4 #256 # train batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = 4 #64 # val batch_size for single GPU
_C.DATA.DATA_PATH = '/dataset/kitti/' # path to dataset
_C.DATA.DATASET = 'kitti' # dataset name
_C.DATA.NUM_WORKERS = 2 # number of data loading threads
_C.DATA.IMG_SIZE = 608
_C.DATA.MULTISCALE = True
_C.DATA.NUM_SAMPLE = None
_C.DATA.RANDOM_PAD = True
_C.DATA.HFLIP_PROB = 0.5
_C.DATA.CUTOUT_PROB = 0
_C.DATA.CUTOUT_NHOLES = 1
_C.DATA.CUTOUT_RATIO = 0.3
_C.DATA.CUTOUT_FILL_VALUE = 0.

# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'YOLO3D-YOLOX'
_C.MODEL.NAME = 'YOLO3D-YOLOX'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 3
_C.MODEL.DROPOUT = 0.1


_C.EVAL = CN()
_C.EVAL.CONF_THRESH = 0.5
_C.EVAL.NMS_THRESH = 0.5
_C.EVAL.IOU_THRESH = 0.5


# training settings
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 3 #34 # ~ 10k steps for 4096 batch size
_C.TRAIN.WEIGHT_DECAY = 0.01 #0.3 # 0.0 for finetune
_C.TRAIN.BASE_LR = 0.001 #0.003 for pretrain # 0.03 for finetune
_C.TRAIN.WARMUP_START_LR = 1e-6 #0.0
_C.TRAIN.END_LR = 1e-5
_C.TRAIN.GRAD_CLIP = 1.0
_C.TRAIN.ACCUM_ITER = 1 #1
_C.TRAIN.PRINT_STEP = 5
_C.TRAIN.LOG_DIR = './log/'
_C.TRAIN.MOSASIC = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'warmupcosine'
_C.TRAIN.LR_SCHEDULER.MILESTONES = "30, 60, 90" # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1 # only used in StepLRScheduler

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)  # for adamW
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# misc
_C.SAVE = "./output"
_C.TAG = "default"
_C.SAVE_FREQ = 20 # freq to save chpt
_C.REPORT_FREQ = 10 # freq to logging info
_C.VALIDATE_FREQ = 20 # freq to do validation
_C.SEED = 42
_C.EVAL = False # run evaluation only
_C.LOCAL_RANK = 0
_C.NGPUS = -1
_C.LOG_DIR = './logs'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as infile:
        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('merging config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    """Update config by ArgumentParser
    Args:
        args: ArgumentParser contains options
    Return:
        config: updated config
    """
    if args.cfg:
        _update_config_from_file(config, args.cfg)
    config.defrost()
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.ngpus:
        config.NGPUS = args.ngpus
    if args.eval:
        config.EVAL = True
        config.DATA.BATCH_SIZE_EVAL = args.batch_size
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.backbone:
        config.MODEL.BACKBONE = args.backbone
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.last_epoch:
        config.MODEL.LAST_EPOCH = args.last_epoch

    #config.freeze()
    return config


def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config