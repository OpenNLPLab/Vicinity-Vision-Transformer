# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import torch.nn.functional as F
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
import utils
import os
import pdb
import os

import inspect
import numpy as np
from torchvision import transforms


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False,
                    distributed=False,
                    test=False,
    ):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # try:
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            outputs = outputs.float() # back to fp32
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if distributed:
            torch.distributed.barrier()
            
        if test:
            break
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print(images.shape)

        # compute outputs
        # with torch.cuda.amp.autocast(enabled = False):
        with torch.cuda.amp.autocast():
            output = model(images)
            output = output.float() # back to fp32
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def generate_cam(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    count = 0
    img_transform = transforms.Compose([transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])


    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        count += 1

        batch_size = images.shape[0]
        # compute outputs
        # with torch.cuda.amp.autocast(enabled = False):
        with torch.cuda.amp.autocast():
            output = model(images)
            output = output.float() # back to fp32

            for batch in range(batch_size):
                cur_image = images[batch,:,:,:]

                rgb_img = img_transform(cur_image)
                rgb_img = rgb_img.detach().cpu().numpy().squeeze()
                rgb_img = rgb_img * 255
                rgb_img = rgb_img.astype(np.uint8)

                cur_target = target[batch]
                # print(cur_target)
                one_hot = np.zeros((1, 1000), dtype=np.float32)
                one_hot[0, cur_target] = 1
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)

                cur_output = output[batch]
                one_hot = (one_hot.cuda() * cur_output)
                # print(one_hot)
                one_hot = torch.sum(one_hot)
                # print(one_hot)

                model.zero_grad()
                one_hot.backward(retain_graph=True)
                cam = model.module.grad_cam(batch, start_layer=0)
                print(cam.shape)
                cam = cam.reshape(7, 7)
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (224, 224), mode='bilinear', align_corners=True)
                cam = cam.detach().cpu().numpy()
                norm_cam = cam / (np.max(cam, (0,1), keepdims=True) + 1e-5)
                print(norm_cam.shape)

                # cv2.imwrite('./cam_output/pvt_v2_b1/{}_{}.png'.format(count, batch), norm_cam[0,0,:,:]*255)


            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def generate_grad_cam(data_loader, model, device, methods):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    count = 0
    img_transform = transforms.Compose([transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    

    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        count += 1

        batch_size = images.shape[0]
        # compute outputs
        # with torch.cuda.amp.autocast(enabled = False):
        with torch.cuda.amp.autocast():
            output = model(images)
            output = output.float() # back to fp32

            for batch in range(batch_size):
                cur_image = images[batch,:,:,:]

                rgb_img = img_transform(cur_image)
                rgb_img = rgb_img.detach().cpu().numpy().squeeze()
                rgb_img = rgb_img * 255
                rgb_img = rgb_img.astype(np.uint8)

                cur_target = target[batch]
                # print(cur_target)
                one_hot = np.zeros((1, 1000), dtype=np.float32)
                one_hot[0, cur_target] = 1
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)

                cur_output = output[batch]
                one_hot = (one_hot.cuda() * cur_output)
                # print(one_hot)
                one_hot = torch.sum(one_hot)
                # print(one_hot)

                model.zero_grad()
                one_hot.backward(retain_graph=True)
                cam = model.module.grad_cam(batch, start_layer=0)
                print(cam.shape)
                cam = cam.reshape(7, 7)
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (224, 224), mode='bilinear', align_corners=True)
                cam = cam.detach().cpu().numpy()
                norm_cam = cam / (np.max(cam, (0,1), keepdims=True) + 1e-5)
                print(norm_cam.shape)
                
                # cv2.imwrite('./cam_output/pvt_v2_b1/{}_{}.png'.format(count, batch), norm_cam[0,0,:,:]*255)


            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



