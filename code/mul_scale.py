import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import ImageFolder
from barbar import Bar
import numpy as np
import time
import copy
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.metrics import geometric_mean_score

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class multi_scale_Reso(nn.Module):
    """Multiple resolution inputs networks using pretraind Resnet50 as the features extractor 
    """
    def __init__(self, use_pretrained= True, num_classes= 102, dropout= 0.0):
        """Creates a multiple resolution inputs model instance.

        Args:
            num_classes (integer): The number of classes
            use_pretrained (bool): Use ImageNet pre-trained backbone feature extractor
            dropout (float): Dropout rate
        """
        super(multi_scale_Reso, self).__init__()

        self.res_branch = models.resnet50(pretrained= use_pretrained)
        input_lentent = self.res_branch.fc.in_features
        self.res_branch.fc = None

        # 112 input
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size= 7, stride= 1, padding= 3,
                               bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 224 input
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size= 7, stride= 2, padding= 3,
                               bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Classifier
        self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(input_lentent, num_classes))

    def forward(self, x1, x2):
        # branch 1
        x1 = self.conv1(x1) 
        x2 = self.conv2(x2)

        x1 = torch.cat((x1, x2), dim=1)

        x1 = self.res_branch.layer1(x1) # 56 128
        x1 = self.res_branch.layer2(x1) # 28
        x1 = self.res_branch.layer3(x1) # 18
        x1 = self.res_branch.layer4(x1) # 7

        x1 = self.res_branch.avgpool(x1)
        x = torch.flatten(x1, 1)

        x = self.fc(x)

        return x
 
class mulinput_ImageFolder(ImageFolder):
    """Multi-resolution inputs dataloader"""
    def __init__(self, root, input_size2 = 112, transform = None, target_transform = None,
            is_valid_file = None):
        super(mulinput_ImageFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform,
                                            is_valid_file=is_valid_file)

        self.input_size2 = input_size2
        self.tf_resize = transforms.Resize((self.input_size2, self.input_size2))
        self.last_tf = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        resized_sample = self.tf_resize(sample)

        sample = self.last_tf(sample)
        resized_sample = self.last_tf(resized_sample)

        return sample, resized_sample, target

    def __len__(self):
        return len(self.samples)

def mulinput_train_model(model, dataloaders, criterion, optimizer, scheduler= None, num_epochs=25
                , is_save_checkpoint= False, device= 'cuda', checkpointFn= None, is_load_checkpoint= False):
    """The function use for training the given model
    
    Args:
        model (nn.Module): Model instance
        dataloader (dictionary): Dictionary contain train and validate dataloader instance
        criterion: Pytorch's loss function instance
        optimizer: Pytorch's optimizer instace
        schedualer (optional): Pytorch's schedualer instnace
        is_inception (bool): This must set to True if the model is an inception model
        is_save_checkpoint (bool, optional): Enable save checkpoint
        device (str) : Device use for training, defualt : 'cuda'
        checkpointFn (str): Checkpoint directory for loading
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

            for input1, input2, labels in Bar(dataloaders[phase]):
                input1 = input1.to(device)
                input2 = input2.to(device)

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input1, input2)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * input1.size(0)
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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate_mulinput_model(model, testloader, path_weight_dict= None, device= 'cuda', model_hyper = {}):
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
            images1, images2, labels = data
            images1 = images1.to(device)
            images2 = images2.to(device)

            temp_outputs = model(images1, images2)
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

        path_folder = os.path.join(os.getcwd(), model_hyper['exp_name'], model_hyper['model_name'] + '_torch')
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

def mulinput_model_predict_prob(model, dataloader, path_weight_dict= None, device= 'cuda', return_labels= False):
    """
    This function use for evaluate predict probability for given model, return torch.tensor

    Args:
        model (nn.Module): Model instance
        dataloader : Pytorch's test dataloader instance
        path_weight_dict (str, optional): Model's weights directory to evaluation
        device (str): Device use for evaluating 
        return_labels (bool): Return the labels
    """
    total = 0
    outputs = None
    model.to(device)
    lables_out = None

    if path_weight_dict != None:
        model.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))

    model.eval()
    with torch.no_grad():
        for data in Bar(dataloader):
            images1, images2, labels = data
            images1 = images1.to(device)
            images2 = images2.to(device)

            temp_outputs = model(images1, images2)
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

        outputs = outputs.to('cpu')

    if return_labels:
        return outputs, labels
    return outputs