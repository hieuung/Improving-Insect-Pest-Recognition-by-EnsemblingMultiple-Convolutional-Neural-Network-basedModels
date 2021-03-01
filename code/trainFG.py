"""This code is to from https://github.com/ZF4444/MMAL-Net"""

#coding=utf-8
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    end_epoch, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, channels, weight_path
from utils.train_model import train
from networks.model import MainNet
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-data", "--dataset", required=True, help= "Choose dataset: IP102, D0 and Chengdata")
ap.add_argument("-dv", "--device", required= True, help= "Device type")

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
args = vars(ap.parse_args())

def main():
    dataset_info = {
    'IP102' : 102,
    'D0' : 40
    }
    dataset_name = args['dataset']
    num_classes = dataset_info[dataset_name]
    if dataset_name == 'D0':
        dataset_name = 'unzip_D0'
    device = args['device']
    
    train_root_path = os.path.join(os.getcwd(), dataset_name, 'train')
    valid_root_path = os.path.join(os.getcwd(), dataset_name, 'valid')

    data_transforms = {
            'train': transforms.Compose([
               transforms.Resize(480),
                transforms.RandomCrop(input_size),             
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(480),
                transforms.CenterCrop(input_size),               
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    #加载数据
    train_set = ImageFolder(root= train_root_path, transform= data_transforms['train']) 
    valid_set = ImageFolder(root= valid_root_path, transform= data_transforms['val'])

    trainloader = DataLoader(train_set, batch_size= batch_size, shuffle= True, 
                            num_workers= 8, pin_memory=True)

    validloader = DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
                            num_workers= 8, pin_memory= True)


    #定义模型
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    #设置训练参数
    criterion = nn.CrossEntropyLoss()

    parameters = model.parameters()

    # #加载checkpoint
    save_path = os.path.join(os.getcwd(), model_path, model_name + '_' + dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpointFn = os.path.join(save_path, 'checkpoint.pt')
    
    lr = init_lr
    start_epoch = 0

    # define optimizers
    optimizer = torch.optim.SGD(parameters, lr= lr, momentum= 0.9, weight_decay= weight_decay)

    model = model.to(device)  # 部署在GPU

    scheduler = MultiStepLR(optimizer, milestones= lr_milestones, gamma= lr_decay_rate)

    # 保存config参数信息
    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    # 开始训练
    model = train(model=model,
          trainloader=trainloader,
          testloader=validloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          checkpointFn=checkpointFn,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_path=save_path,
          is_load_checkpoint= False)

    torch.save(model.state_dict(), os.path.join(weight_path, model_name + '_' + dataset_name + '.pt'))

if __name__ == '__main__':
    main()