import os
import math
import time
import argparse
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.yolo3dx import YOLO3DX

from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint, save_best_checkpoint
from utils.train_utils import to_python_float
from utils.misc import AverageMeter
from configs import get_config, update_config
from evaluate import evaluate_mAP

parser = argparse.ArgumentParser('YOLO3D-YOLOX')
parser.add_argument('-cfg', type=str, default=None)
parser.add_argument('-dataset', type=str, default=None)
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-data_path', type=str, default=None)
parser.add_argument('-backbone', type=str, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-pretrained', type=str, default=None)
parser.add_argument('-resume', type=str, default=None)
parser.add_argument('-last_epoch', type=int, default=None)
parser.add_argument('-eval', action='store_true')
arguments = parser.parse_args()

configs = get_config()
configs = update_config(configs, arguments)

def train_one_epoch(dataloader,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    configs,
                    tb_writer):
    batch_time = AverageMeter('Time',':6.3f')
    data_time  = AverageMeter('Data',':6.3f')
    losses     = AverageMeter('Loss',':.4e')

    num_iters_per_epoch = len(dataloader)
    model.train()
    start_time = time.time()
    for batch_idx, batch_data  in enumerate(tqdm(dataloader)):
        # init some statistics code
        data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * epoch + batch_idx + 1
        batch_size = imgs.size(0)

        # transfer the data from cpu memory to gpu memory
        targets = targets.to('cuda', non_blocking=True)
        imgs = imgs.to('cuda', non_blocking=True)
        # forward the model
        loss = model(imgs, targets)
        total_loss = loss["total_loss"]
        iou_loss   = loss["iou_loss"]
        obj_loss   = loss["conf_loss"]
        cls_loss   = loss["cls_loss"]
        yaw_loss   = loss["yaw_loss"]

        # backward the model
        total_loss.backward()
        # optimizer's update
        if global_step % configs.TRAIN.WARMUP_EPOCHS == 0:
            optimizer.step()
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)
            # zero the parameter gradients
            optimizer.zero_grad()
        if global_step % configs.TRAIN.PRINT_STEP == 0:
            print("In Step ", global_step, " / ", num_iters_per_epoch * (epoch + 1), " Now avg loss = ", losses.avg,
                  " iou_loss = ", iou_loss.cpu().detach().numpy(), " obj_loss = ", obj_loss.cpu().detach().numpy(), 
                  " cls_loss = ", cls_loss.cpu().detach().numpy()," yaw_loss = ", yaw_loss.cpu().detach().numpy())
            #print("Now the total_loss = ", total_loss)
            #print("Now the loss = ", loss)

        # update the static information
        losses.update(to_python_float(total_loss.data), batch_size)
        batch_time.update(time.time() - start_time)




        # upadate the start time
        start_time = time.time()



def main():
    global_ap = None
    tb_writer = SummaryWriter(log_dir=os.path.join(configs.TRAIN.LOG_DIR, 'tensorboard'))

    model = YOLO3DX(configs)
    model = model.cuda()

    if configs.MODEL.PRETRAINED is not None:
        assert os.path.isfile(configs.MODEL.PRETRAINED), \
            "No checkpoint found at '{}'".format(configs.MODEL.PRETRAINED)
        model.load_state_dict(torch.load(configs.MODEL.PRETRAINED))
        print("Loaded pretrained model at {} ".format(configs.MODEL.PRETRAINED))

    optimizer    = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    start_epoch  = 0
    if configs.MODEL.RESUME is not None:
        utils_path = configs.MODEL.RESUME.replace('Model_', 'Utils_')
        assert os.path.isfile(configs.MODEL.RESUME), \
            "No checkpoint found at '{}'".format(configs.MODEL.RESUME)
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        # load resume model
        model.load_state_dict(torch.load(configs.MODEL.RESUME))
        # load optimizer and lr_scheduler
        utils_state_dict = torch.load(utils_path, map_location='cuda:0')
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        start_epoch = utils_state_dict['epoch'] + 1
        print("Resume training model from checkpoint {}".format(configs.MODEL.RESUME))

    train_dataloader = create_train_dataloader(configs)
    val_dataloader = create_val_dataloader(configs)

    # epoch training loop. training model, eval model, save best model
    for epoch in range(start_epoch, configs.TRAIN.NUM_EPOCHS + 1):
        print(">>>> Epoch: [{}/{}]".format(epoch, configs.TRAIN.NUM_EPOCHS))

        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, tb_writer)
        if (epoch > 50) and (epoch % configs.SAVE_FREQ == 0):
            #model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            #save_checkpoint("./output/", "yolo3dx", model_state_dict, utils_state_dict, epoch)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs)
            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }
            if global_ap is not None:
                if AP.mean() > global_ap:
                    model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
                    save_best_checkpoint(configs.SAVE, AP.mean(), model_state_dict, utils_state_dict, epoch)
                global_ap = AP.mean()
            else:
                global_ap = AP.mean()
                model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
                save_checkpoint(configs.SAVE, configs.SAVE_FN, model_state_dict, utils_state_dict, epoch)
            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

            if not configs.step_lr_in_epoch:
                lr_scheduler.step()
                if tb_writer is not None:
                    tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    # close the tbwriter
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    main()
