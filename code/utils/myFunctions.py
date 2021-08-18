"""This module contains utility functions and object"""

import os
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import torch.nn as nn
import time
import copy
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.metrics import geometric_mean_score
from barbar import Bar
from PIL import Image, ImageOps
from networks.residual_attention_network import ResidualAttentionModel_56
from networks.fpn import Resnet_fpn

def train_model(model, dataloaders, criterion, optimizer, scheduler= None, num_epochs=25
                , is_inception= False, is_save_checkpoint= False, device= 'cuda', checkpointFn= None, is_load_checkpoint= False):
    """The function use for training the given model
    
    Args:
        model (nn.Module): Model instance
        dataloader (dictionary): Dictionary contain train and validate dataloader instance
        criterion: Pytorch's loss function instance
        optimizer: Pytorch's optimizer instance
        schedualer (optional): Pytorch's schedualer instnace
        is_inception (bool): This must set to True if the model is an inception model
        is_save_checkpoint (bool, optional): Enable save checkpoint
        device (str) : Device use for training, defualt : 'cuda'
        checkpointFn (str): Checkpoint directory for loading and saving
        is_load_checkpoint (bool, optional): Enable load the given checkpoint"""
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    if is_load_checkpoint:
        checkpoint = torch.load(checkpointFn)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ckpepoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        best_acc = 0.0
        ckpepoch = 0

    count2stop = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(ckpepoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in Bar(dataloaders[phase]):
                if len(data) > 2:
                    inputs, _ , labels = data
                    inputs = inputs.to(device)
                else:
                    inputs, labels = data
                    inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if is_save_checkpoint:
                    torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'scheduler_state_dict': scheduler.state_dict(),
                            }, checkpointFn)
                
                count2stop = 0

            elif phase == 'val':
                count2stop += 1

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

                
        if count2stop == 10:
            break

        if scheduler:
            scheduler.step()
            print('lr :', scheduler.get_lr())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    his = {'train_loss': train_loss_history, 
           'train_acc': train_acc_history,
           'val_loss': val_loss_history, 
           'val_acc': val_acc_history}
    return model, his

def accuracy(output, target, topk=(1,)):
    """Computes the precision @k for the specified values of k
    
    Args:
        output (torch.tensor): Model's prediction
        target (torch.tensor): Ground truth
        topk (iterable): Top k accuracies"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def initialize_model(model_name, num_classes, use_pretrained=True, dropout= 0.5, num_fpn_filter= 256):
    """Initialize model
    
    Args:
        model_name (string): Model's name
        num_classes (integer): Number of classes
        use_pretrained (bool): Use pretrained model's weight
        dropout (float): Dropout rate
        num_fpn_filter (integer): Use for Pyramid Feature network only"""

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                      nn.Dropout(dropout),
                      nn.Linear(num_ftrs, num_classes))
        input_size = 224
    elif model_name == "residual-attention":
        model_ft = ResidualAttentionModel_56()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                      nn.Dropout(dropout),
                      nn.Linear(num_ftrs, num_classes))
        input_size = 224
    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)

        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    elif model_name == "fpn":
        model_ft = Resnet_fpn(num_classes= num_classes,num_filters = num_fpn_filter, pretrained= use_pretrained)
        input_size = 224
    else:
        raise Exception('Error, invalid model name')

    return model_ft, input_size

class resizePadding(object):
    """Custom tranform: Resize, pad image to square shape while keep its aspect ratio"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.desired_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.desired_size = output_size

    def __call__(self, im):
        old_size = im.size

        ratio = float(self.desired_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im = im.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", self.desired_size)
        new_im.paste(im, ((self.desired_size[0] - new_size[0])//2,
                            (self.desired_size[1] - new_size[1])//2))
        return new_im

def evaluate_model(model, testloader, path_weight_dict= None, device= 'cuda', model_hyper = {}):
    """
    This function use for evaluating model then write down the report
    
    Args:
        model (nn.Module): Model instance
        testloader : Pytorch's test dataloader instance
        path_weight_dict (str, optional): Model's weights directory to evaluation
        device (str): Device use for evaluating
        model_hyper (dictionary): Model hyperparameters  
    """
    total = 0
    topk=(1, 3, 5)
    y_test = None
    outputs = None
    predicted = None
    model.to(device)

    if path_weight_dict != None:
        model.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))

    model.eval()
    with torch.no_grad():
        for data in Bar(testloader):
            images, labels = data
            images = images.to(device)

            temp_outputs = model(images)
            temp_outputs.to(device)

            if outputs is None:
                outputs = temp_outputs
            else:
                outputs = torch.cat((outputs, temp_outputs))
            total += labels.size(0)
            
            if y_test is None:
                y_test = labels
            else:
                y_test = torch.cat((y_test, labels))

            _, temp_predicted = torch.max(temp_outputs.data, 1)
            if predicted is None:
                predicted = temp_predicted
            else:
                predicted = torch.cat((predicted, temp_predicted))

        predicted = predicted.to('cpu')
        outputs = outputs.to('cpu')

        topKAccuracy = accuracy(outputs, y_test, topk= topk)

        path_folder = os.path.join(os.getcwd(), model_hyper['exp_name'], model_hyper['model_name'] + '_' + model_hyper['dataset'] + '_torch')
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        with open(os.path.join(path_folder, ('Result {s}.txt').format(s = model_hyper['model_name'])), 'w') as f:
            f.write('Number epochs : %d\n' %(model_hyper['num_epochs']))
            f.write('Learning rate(init) : %f\n' %(model_hyper['init_lr']))
            f.write('L2 regularization lambda: %f\n' %(model_hyper['weight_decay']))
            f.write('Dropout rate: %f\n' %(model_hyper['dropout']))
            f.write('\n')
            f.write(str(classification_report(predicted.numpy(), y_test.numpy(), 
                                        digits= np.int64(np.ceil(np.log(total))))))
            for i, k in enumerate(topk):
                f.write('Top %d : %f\n' %(k, topKAccuracy[i]))
            f.write('Geometric mean : %f\n' %(geometric_mean_score(y_test.numpy(), predicted.numpy())))
        for i, k in enumerate(topk):
            print('Top %d : %f' %(k, topKAccuracy[i]))

        print('Report : \n', classification_report(predicted.numpy(), y_test.numpy(), 
                                        digits= np.int64(np.ceil(np.log(total)))))
        print('Geometric mean : %f\n' %(geometric_mean_score(y_test.numpy(), predicted.numpy())))

class myScheduler(object):
    """
    Custom scheduler, specifically use for the Residual Attention networks
    """
    def __init__(self, optimizer, gamma= 0.09999):
        self.optimizer = optimizer
        self.step_sizes = [30,  10]
        self.idx = 0
        self.step_after_change = 1

        self._get_lr_called_within_step = False
        self.gamma = gamma
        self.last_epoch = 0
        self._step_count = 1
        self._last_lr = [self.optimizer.param_groups[0]['lr']]
        self.base_lrs = [self.optimizer.param_groups[0]['lr']]
        self.step_size = self.step_sizes[self.idx]

    def state_dict(self):
        sd = {
            '_get_lr_called_within_step': self._get_lr_called_within_step,
            '_last_lr': self._last_lr,
            '_step_count': self._step_count,
            'base_lrs': self.base_lrs,
            'gamma': self.gamma,
            'last_epoch': self.last_epoch,
            'step_size': self.step_size,
            'step_after_change': self.step_after_change,
            'idx': self.idx
        }
        return sd

    def step(self):
        if self.step_after_change == self.step_size:
            self.optimizer.param_groups[0]['lr'] = self._last_lr[0] * self.gamma
            self._last_lr = [self._last_lr[0] * self.gamma]

            if self.idx < len(self.step_sizes) - 1:
                self.idx += 1
                self.step_size = self.step_sizes[self.idx]

            self.step_after_change = 1
            self.last_epoch = self._step_count
            self._step_count += 1
        else:
            self.last_epoch = self._step_count
            self._step_count += 1
            self.step_after_change += 1

    def load_state_dict(self, state_dict):
        self._get_lr_called_within_step = state_dict['_get_lr_called_within_step']
        self.gamma = state_dict['gamma']
        self.last_epoch = state_dict['last_epoch']
        self._step_count = state_dict['_step_count']
        self._last_lr = state_dict['_last_lr']
        self.base_lrs = state_dict['base_lrs']
        self.step_size = state_dict['step_size']
        self.step_after_change = state_dict['step_after_change']
        self.idx = state_dict['idx']

    def get_lr(self):
        return self._last_lr[0]

    def get_count(self):
        return self._step_count

def model_predict(model, dataloader, path_weight_dict= None, device= 'cuda', return_labels= False):
    """
    This function use for evaluate predict probability for given model, return torch.tensor

    Args:
        model (nn.Module): Model instance
        dataloader : Pytorch's test dataloader instance
        path_weight_dict (str, optional): Model's weights directory to evaluation
        device (str): Device use for evaluating 
        return_labels (bool): Return the labels
    """
    outputs = None
    model.to(device)
    lables_out = None

    if path_weight_dict != None:
        model.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))

    model.eval()
    with torch.no_grad():
        for data in Bar(dataloader):
            images, labels = data
            images = images.to(device)

            temp_outputs = model(images)
            temp_outputs.to(device)

            if outputs is None:
                outputs = temp_outputs
            else:
                outputs = torch.cat((outputs, temp_outputs))
                
            if return_labels:
                if lables_out is None:
                    lables_out = labels
                else:
                    lables_out = torch.cat((lables_out, labels))

        outputs = torch.nn.functional.softmax(outputs.to('cpu'), dim=-1)

    if return_labels:
        return outputs, lables_out
    return outputs

def finegGrained_pred(model, testloader, device= 'cuda', return_labels= False):

    model.eval()
    y_test = None
    outputs = None
    with torch.no_grad():
        for data in Bar(testloader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            raw_logits, local_logits, _ = model(x, 'test', device)[-3:]
            local_logits.to(device)
            raw_logits.to(device)

            logits = (torch.nn.functional.softmax(local_logits, dim=-1) 
            + torch.nn.functional.softmax(raw_logits, dim=-1)) / 2
            logits.to(device)

            if outputs is None:
                outputs = logits
            else:
                outputs = torch.cat((outputs, logits))
                
            if y_test is None:
                y_test = y
            else:
                y_test = torch.cat((y_test, y))

        outputs = outputs.to('cpu')
        y_test = y_test.to('cpu')

    if return_labels:
        return outputs, y_test
    return outputs