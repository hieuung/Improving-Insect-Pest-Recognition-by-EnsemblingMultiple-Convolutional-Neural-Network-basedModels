"""This code is refer to https://github.com/ZF4444/MMAL-Net"""

import torch
from barbar import Bar
import os
from tensorboardX import SummaryWriter
import numpy as np
from config import coordinates_cat, proposalN, vis_num
from utils.cal_iou import calculate_iou
from utils.vis import image_with_boxes

def eval(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print('Evaluating')

    raw_loss_sum = 0
    local_loss_sum = 0
    windowscls_loss_sum = 0
    total_loss_sum = 0
    iou_corrects = 0
    raw_correct = 0
    local_correct = 0

    with torch.no_grad():
        for i, data in enumerate(Bar(testloader)):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            _, proposalN_windows_logits, indices, \
            _, coordinates, raw_logits, local_logits, local_imgs = model(images, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                                        labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            total_loss = raw_loss + local_loss + windowscls_loss

            raw_loss_sum += raw_loss.item()
            local_loss_sum += local_loss.item()
            windowscls_loss_sum += windowscls_loss.item()

            total_loss_sum += total_loss.item()

            # correct num
            # raw
            pred = raw_logits.max(1, keepdim=True)[1]
            raw_correct += pred.eq(labels.view_as(pred)).sum().item()
            # local
            pred = local_logits.max(1, keepdim=True)[1]
            local_correct += pred.eq(labels.view_as(pred)).sum().item()

            if i == 0:
            # object branch tensorboard
                indices_ndarray = indices[:vis_num,:proposalN].cpu().numpy()
                with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment= status + 'object') as writer:
                    cat_imgs = []
                    for j, indice_ndarray in enumerate(indices_ndarray):
                        img = image_with_boxes(local_imgs[j], coordinates_cat[indice_ndarray])
                        cat_imgs.append(img)
                    cat_imgs = np.concatenate(cat_imgs, axis=1)
                    writer.add_images(status + '/' + 'object image with windows', cat_imgs, epoch, dataformats='HWC')

    raw_loss_avg = raw_loss_sum / (i+1)
    local_loss_avg = local_loss_sum / (i+1)
    windowscls_loss_avg = windowscls_loss_sum / (i+1)
    total_loss_avg = total_loss_sum / (i+1)

    raw_accuracy = raw_correct / len(testloader.dataset)
    local_accuracy = local_correct / len(testloader.dataset)

    return raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
           local_loss_avg