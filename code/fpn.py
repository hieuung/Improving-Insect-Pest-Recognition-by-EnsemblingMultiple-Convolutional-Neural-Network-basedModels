# import os
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import models
# from torchvision.datasets import ImageFolder
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import torch.nn as nn
# import torch.optim as optim
# import time
# import copy
# from PIL import Image, ImageOps

def convert_to_inplace_relu(model):
  for m in model.modules():
    if isinstance(m, nn.ReLU):
      m.inplace = True

class Resnet_fpn(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral
       connections. Can be used as feature extractor for object detection
       or segmentation.
    """

    def __init__(self, num_classes=102, num_filters=256, pretrained=True):
        """Creates a `FPN` model instance.

        Args:
            num_filters (integer): The number of filters in each output pyramid level
            pretrained (bool): Use ImageNet pre-trained backbone feature extractor
            num_input_channels (integer): Number fo input channels
        """

        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        self.resnet = models.resnet50(pretrained=pretrained)
        num_bottleneck_filters = 2048

        self.lateral4 = Conv1x1(num_bottleneck_filters, num_filters)
        self.lateral3 = Conv1x1(num_bottleneck_filters // 2, num_filters)
        self.lateral2 = Conv1x1(num_bottleneck_filters // 4, num_filters)
        self.lateral1 = Conv1x1(num_bottleneck_filters // 8, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)

        self.fc = nn.Linear(num_filters * 4, num_classes)

    def forward_s4(self, enc0):
        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.functional.interpolate(map4, scale_factor=2,
            mode="nearest")
        map2 = lateral2 + nn.functional.interpolate(map3, scale_factor=2,
            mode="nearest")
        map1 = lateral1 + nn.functional.interpolate(map2, scale_factor=2,
            mode="nearest")
        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        map1, map2, map3, map4 = self.forward_s4(enc0)

        x1 = self.resnet.avgpool(map1)
        x1 = torch.flatten(x1, 1)
        x2 = self.resnet.avgpool(map2)
        x2 = torch.flatten(x2, 1)  
        x3 = self.resnet.avgpool(map3)
        x3 = torch.flatten(x3, 1)
        x4 = self.resnet.avgpool(map4)
        x4 = torch.flatten(x4, 1)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.fc(x)

        return x

class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x)

class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        return self.block(x)

# def train_model(model, dataloaders, criterion, optimizer, scheduler= None, num_epochs=25
#                 , is_inception= False, is_save_checkpoint= False, device= 'cuda', checkpointFn= None, is_load_checkpoint= False):
#     """The function use for training th given model
    
#     Args:
#         model (nn.Module): Model instance
#         dataloader (dictionary): Dictionary contain train and validate dataloader instance
#         criterion: Pytorch's loss function instance
#         optimizer: Pytorch's optimizer object
#         schedualer (optional): Pytorch's schedualer instnace
#         is_inception (bool): This must set to True if the model is an inception model
#         is_save_checkpoint (bool, optional): Enable save checkpoint
#         device (str) : Device use for training, defualt : 'cuda'
#         checkpointFn (str): Checkpoint directory for loading
#         is_load_checkpoint (bool, optional): Enable load the given checkpoint"""
    
#     since = time.time()

#     val_acc_history = []
#     train_acc_history = []
#     val_loss_history = []
#     train_loss_history = []

#     if is_load_checkpoint:
#         checkpoint = torch.load(checkpointFn)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         ckpepoch = checkpoint['epoch']
#         best_acc = checkpoint['best_acc']
#     else:
#         best_acc = 0.0
#         ckpepoch = 0

#     count2stop = 0
#     best_model_wts = copy.deepcopy(model.state_dict())
#     for epoch in range(ckpepoch, num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)

#                     _, preds = torch.max(outputs, 1)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
                        
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 if is_save_checkpoint:
#                     torch.save({'epoch': epoch,
#                             'model_state_dict': model.state_dict(),
#                             'optimizer_state_dict': optimizer.state_dict(),
#                             'best_acc': best_acc,
#                             'scheduler_state_dict': scheduler.state_dict(),
#                             }, checkpointFn)
                
#                 count2stop = 0

#             elif phase == 'val':
#                 count2stop += 1

#             if phase == 'val':
#                 val_loss_history.append(epoch_loss)
#                 val_acc_history.append(epoch_acc)
#             if phase == 'train':
#                 train_loss_history.append(epoch_loss)
#                 train_acc_history.append(epoch_acc)

                
#         if count2stop == 10:
#             break

#         if scheduler:
#             scheduler.step()
#             print('lr :', scheduler.get_lr())

#         print()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     model.load_state_dict(best_model_wts)
#     his = {'train_loss': train_loss_history, 
#            'train_acc': train_acc_history,
#            'val_loss': val_loss_history, 
#            'val_acc': val_acc_history}
#     return model, his

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision @k for the specified values of k
    
#     Args:
#         output (torch.tensor): Model's prediction
#         target (torch.tensor): Ground truth
#         topk (iterable): Top k accuracies"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

# def evaluate_model(model, testloader, path_weight_dict= None, device= 'cuda', model_hyper = {}):
#     """
#     This function use for evaluating model then write down the report
    
#     Args:
#         model (nn.Module): Model instance
#         testloader : Pytorch's test dataloader instance
#         path_weight_dict (str, optional): Model's weights directory to evaluation
#         device (str): Device use for evaluating
#         model_hyper (dictionary): Model hyperparameters  
#     """
#     total = 0
#     topk=(1, 3, 5)
#     y_test = None
#     outputs = None
#     predicted = None
#     model.to(device)

#     if path_weight_dict != None:
#         model.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))

#     model.eval()
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)

#             temp_outputs = model(images)
#             temp_outputs.to(device)

#             if outputs is None:
#                 outputs = temp_outputs
#             else:
#                 outputs = torch.cat((outputs, temp_outputs))
#             total += labels.size(0)
            
#             if y_test is None:
#                 y_test = labels
#             else:
#                 y_test = torch.cat((y_test, labels))

#             _, temp_predicted = torch.max(temp_outputs.data, 1)
#             if predicted is None:
#                 predicted = temp_predicted
#             else:
#                 predicted = torch.cat((predicted, temp_predicted))

#         predicted = predicted.to('cpu')
#         outputs = outputs.to('cpu')
#         topKAccuracy = accuracy(outputs, y_test, topk= topk)

#         path_folder = os.path.join(os.getcwd(), model_hyper['exp_name'], model_hyper['model_name'] + '_torch')
#         if not os.path.exists(path_folder):
#             os.makedirs(path_folder)
#         with open(os.path.join(path_folder, ('Result {s}.txt').format(s = model_hyper['model_name'])), 'w') as f:
#             f.write('Number epochs : %d\n' %(model_hyper['num_epochs']))
#             f.write('Learning rate(init) : %f\n' %(model_hyper['init_lr']))
#             f.write('L2 regularization lambda: %f\n' %(model_hyper['weight_decay']))
#             f.write('Dropout rate: %f\n' %(model_hyper['dropout']))
#             f.write('\n')
#             f.write(str(classification_report(predicted.numpy(), y_test.numpy(), 
#                                         digits= np.int64(np.ceil(np.log(total))))))
#             for i, k in enumerate(topk):
#                 f.write('Top %d : %f\n' %(k, topKAccuracy[i]))
#             f.write('Geometric mean : %f\n' %(geometric_mean_score(y_test.numpy(), predicted.numpy())))
#         for i, k in enumerate(topk):
#             print('Top %d : %f' %(k, topKAccuracy[i]))

#         print('Report : \n', classification_report(predicted.numpy(), y_test.numpy(), 
#                                         digits= np.int64(np.ceil(np.log(total)))))
#         print('Geometric mean : %f\n' %(geometric_mean_score(y_test.numpy(), predicted.numpy())))