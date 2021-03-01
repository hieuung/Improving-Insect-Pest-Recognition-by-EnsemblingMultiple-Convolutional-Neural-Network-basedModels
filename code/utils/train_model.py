"""This code is refer to https://github.com/ZF4444/MMAL-Net"""

import os
import copy
import glob
import torch
from barbar import Bar
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset
from utils.eval_model import eval

def train(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          checkpointFn,
          start_epoch,
          end_epoch,
          save_path,
          is_load_checkpoint):

    if is_load_checkpoint:
        checkpoint = torch.load(checkpointFn)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        best_acc = 0.0
        start_epoch = 0

    count2stop = 0
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(Bar(trainloader)):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = raw_loss + local_loss + windowscls_loss

            total_loss.backward()

            optimizer.step()

        print('lr :', scheduler.get_lr())
        scheduler.step()

        # evaluation every epoch
        if eval_trainset:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:

                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        # eval valid
        raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
        local_loss_avg\
            = eval(model, testloader, criterion, 'valid', save_path, epoch)

        print(
            'Test set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                100. * raw_accuracy, 100. * local_accuracy))

        # tensorboard
        with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='valid') as writer:
            writer.add_scalar('valid/raw_accuracy', raw_accuracy, epoch)
            writer.add_scalar('valid/local_accuracy', local_accuracy, epoch)
            writer.add_scalar('valid/raw_loss_avg', raw_loss_avg, epoch)
            writer.add_scalar('valid/local_loss_avg', local_loss_avg, epoch)
            writer.add_scalar('valid/windowscls_loss_avg', windowscls_loss_avg, epoch)
            writer.add_scalar('valid/total_loss_avg', total_loss_avg, epoch)

        if local_accuracy > best_acc:
            best_acc = local_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            count2stop = 0
        else:
            count2stop += 1

        torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'scheduler_state_dict': scheduler.state_dict(),
                            }, checkpointFn)

        if count2stop == 10:
            break

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model