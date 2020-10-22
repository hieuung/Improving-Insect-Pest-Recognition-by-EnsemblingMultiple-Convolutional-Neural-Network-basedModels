from ensamble import *
from mul_scale import multi_scale_Reso, mulinput_train_model, evaluate_mulinput_model, mulinput_ImageFolder
from myFunctions import initialize_model, train_model, myScheduler
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os

if __name__ == "__main__":
    ###########################################################
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = 'cuda'
    n_classes = 102
    batch_size = 64
    num_epochs = 100
    exp_name = 'insect_recognition'
    input_size = 224
    save_model = True
    is_save_checkpoint = True
    load_checkpoint = False

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size)                
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])             
        ]),
    }

    train_set = mulinput_ImageFolder(root= 'train', transform= data_transforms['train'])

    set12, set34 = train_test_split(list(range(len(train_set))), test_size= 0.5, 
                          stratify= train_set.targets, shuffle= True, random_state= 42)

    set1, set2 = train_test_split(set12, test_size= 0.5, 
                            stratify= np.array(train_set.targets)[np.array(set12)], shuffle= True, random_state= 42)

    set3, set4 = train_test_split(set34, test_size= 0.5, 
                            stratify= np.array(train_set.targets)[np.array(set34)], shuffle= True, random_state= 42)

    datasets = {
        'set1': Subset(train_set, set1), 
        'set2': Subset(train_set, set2), 
        'set3': Subset(train_set, set3), 
        'set4': Subset(train_set, set4)
    }
    
    valid_set = ImageFolder(root= 'valid', transform= data_transforms['val'])

    # RESNET50
    init_lr = 1e-4
    weight_decay = 0.00000
    dropout = 0.5
    optimizer = 'Adam'
    scheduler = 'expDecay'
    use_pretrained = True
    model_name = 'resnet'
    checkpoint = 'checkpoint_' + model_name + '.pt'

    model_hyper = {
        'batch_size' : batch_size,
        'num_epochs' : num_epochs,
        'init_lr' : init_lr,
        'weight_decay' : weight_decay,
        'dropout' : dropout,
        'model_name' : model_name,
        'exp_name' : exp_name
    }

    model_ft, _ = initialize_model(model_name, n_classes, 
                    use_pretrained= use_pretrained, dropout= dropout)

    params_to_update = nn.ParameterList(model_ft.parameters())

    opt_switcher = {
        'sgd': optim.SGD(params_to_update, lr= init_lr, momentum= 0.9, weight_decay= weight_decay),
        'adam': optim.Adam(params_to_update, lr= init_lr, betas= (0.9, 0.999),
                                eps= 1e-08, weight_decay= weight_decay)
    }
    optimizer_ft = opt_switcher.get(optimizer.lower(), None)
    
    sch_switcher = {
        'expdecay': optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma= 0.96),
        'steplr': optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma= 0.09999, last_epoch=-1),
        'myscheduler': myScheduler(optimizer_ft, gamma= 0.09999)
    }
    scheduler_ft = sch_switcher.get(scheduler.lower(), None)

    criterion = nn.CrossEntropyLoss()

    datasets_dict = {'train': DataLoader(datasets['set1'], batch_size= batch_size, shuffle= True, 
                            num_workers= 8, pin_memory= True), 
                    'val': DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
                            num_workers= 8, pin_memory= True)}

    model_ft, _ = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                        scheduler= scheduler_ft, num_epochs= num_epochs,
                                        is_inception= model_name == 'inception', checkpointFn= checkpoint
                                        , device= device, is_save_checkpoint= is_save_checkpoint
                                        ,is_load_checkpoint= load_checkpoint)
    print("RESNET50 done")

    # # FPN
    # init_lr = 1e-4
    # weight_decay = 0.00000
    # dropout = 0.5
    # optimizer = 'Adam'
    # scheduler = 'expDecay'
    # use_pretrained = True
    # model_name = 'fpn'
    # checkpoint = 'checkpoint_' + model_name + '.pt'

    # model_hyper = {
    #     'batch_size' : batch_size,
    #     'num_epochs' : num_epochs,
    #     'init_lr' : init_lr,
    #     'weight_decay' : weight_decay,
    #     'dropout' : dropout,
    #     'model_name' : model_name,
    #     'exp_name' : exp_name
    # }

    # model_ft, _ = initialize_model(model_name, n_classes, 
    #                 use_pretrained= use_pretrained, dropout= dropout)

    # params_to_update = nn.ParameterList(model_ft.parameters())

    # opt_switcher = {
    #     'sgd': optim.SGD(params_to_update, lr= init_lr, momentum= 0.9, weight_decay= weight_decay),
    #     'adam': optim.Adam(params_to_update, lr= init_lr, betas= (0.9, 0.999),
    #                             eps= 1e-08, weight_decay= weight_decay)
    # }
    # optimizer_ft = opt_switcher.get(optimizer.lower(), None)
    
    # sch_switcher = {
    #     'expdecay': optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma= 0.96),
    #     'steplr': optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma= 0.09999, last_epoch=-1),
    #     'myscheduler': myScheduler(optimizer_ft, gamma= 0.09999)
    # }
    # scheduler_ft = sch_switcher.get(scheduler.lower(), None)

    # criterion = nn.CrossEntropyLoss()

    # datasets_dict = {'train': DataLoader(datasets['set2'], batch_size= batch_size, shuffle= True, 
    #                         num_workers= 8, pin_memory= True), 
    #                 'val': DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
    #                         num_workers= 8, pin_memory= True)}

    # model_ft, _ = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
    #                                     scheduler= scheduler_ft, num_epochs= num_epochs,
    #                                     is_inception= model_name == 'inception', checkpointFn= checkpoint
    #                                     , device= device, is_save_checkpoint= is_save_checkpoint
    #                                     ,is_load_checkpoint= load_checkpoint)
    # print("Feature Pyramid done")

    # # residual-attention
    # init_lr = 0.1
    # weight_decay = 0.000
    # dropout = 0.0
    # optimizer = 'SGD'
    # scheduler = 'myScheduler'
    # model_name = 'residual-attention'
    # checkpoint = 'checkpoint_' + model_name + '.pt'

    # model_hyper = {
    #     'batch_size' : batch_size,
    #     'num_epochs' : num_epochs,
    #     'init_lr' : init_lr,
    #     'weight_decay' : weight_decay,
    #     'dropout' : dropout,
    #     'model_name' : model_name,
    #     'exp_name' : exp_name
    # }

    # model_ft, _ = initialize_model(model_name, n_classes, 
    #                 use_pretrained= use_pretrained, dropout= dropout)

    # params_to_update = nn.ParameterList(model_ft.parameters())

    # opt_switcher = {
    #     'sgd': optim.SGD(params_to_update, lr= init_lr, momentum= 0.9, weight_decay= weight_decay),
    #     'adam': optim.Adam(params_to_update, lr= init_lr, betas= (0.9, 0.999),
    #                             eps= 1e-08, weight_decay= weight_decay)
    # }
    # optimizer_ft = opt_switcher.get(optimizer.lower(), None)
    
    # sch_switcher = {
    #     'expdecay': optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma= 0.96),
    #     'steplr': optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma= 0.09999, last_epoch=-1),
    #     'myscheduler': myScheduler(optimizer_ft, gamma= 0.09999)
    # }
    # scheduler_ft = sch_switcher.get(scheduler.lower(), None)

    # criterion = nn.CrossEntropyLoss()

    # datasets_dict = {'train': DataLoader(datasets['set3'], batch_size= batch_size, shuffle= True, 
    #                         num_workers= 8, pin_memory= True), 
    #                 'val': DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
    #                         num_workers= 8, pin_memory= True)}

    # model_ft, _ = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
    #                                     scheduler= scheduler_ft, num_epochs= num_epochs,
    #                                     is_inception= model_name == 'inception', checkpointFn= checkpoint
    #                                     , device= device, is_save_checkpoint= is_save_checkpoint
    #                                     ,is_load_checkpoint= load_checkpoint)
    # print("Residual-attention done")

    # # multi-reso
    # init_lr = 1e-4
    # weight_decay = 0.00000
    # dropout = 0.5
    # optimizer = 'Adam'
    # scheduler = 'expDecay'
    # use_pretrained = True
    # model_name = 'multi-reso'
    # checkpoint = 'checkpoint_' + model_name + '.pt'

    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.RandomCrop(input_size)                
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(input_size) 
    #     ])
    # }

    # model_hyper = {
    #     'batch_size' : batch_size,
    #     'num_epochs' : num_epochs,
    #     'init_lr' : init_lr,
    #     'weight_decay' : weight_decay,
    #     'dropout' : dropout,
    #     'model_name' : model_name,
    #     'exp_name' : exp_name
    # }

    # model_ft, _ = initialize_model(model_name, n_classes, 
    #                 use_pretrained= use_pretrained, dropout= dropout)

    # params_to_update = nn.ParameterList(model_ft.parameters())

    # opt_switcher = {
    #     'sgd': optim.SGD(params_to_update, lr= init_lr, momentum= 0.9, weight_decay= weight_decay),
    #     'adam': optim.Adam(params_to_update, lr= init_lr, betas= (0.9, 0.999),
    #                             eps= 1e-08, weight_decay= weight_decay)
    # }
    # optimizer_ft = opt_switcher.get(optimizer.lower(), None)
    
    # sch_switcher = {
    #     'expdecay': optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma= 0.96),
    #     'steplr': optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma= 0.09999, last_epoch=-1),
    #     'myscheduler': myScheduler(optimizer_ft, gamma= 0.09999)
    # }
    # scheduler_ft = sch_switcher.get(scheduler.lower(), None)

    # criterion = nn.CrossEntropyLoss()

    # valid_set = mulinput_ImageFolder(root= 'valid', transform= data_transforms['val'])
    # datasets_dict = {'train': DataLoader(datasets['set4'], batch_size= batch_size, shuffle= True, 
    #                         num_workers= 8, pin_memory= True), 
    #                 'val': DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
    #                         num_workers= 8, pin_memory= True)}

    # model_ft, _ = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
    #                                     scheduler= scheduler_ft, num_epochs= num_epochs,
    #                                     is_inception= model_name == 'inception', checkpointFn= checkpoint
    #                                     , device= device, is_save_checkpoint= is_save_checkpoint
    #                                     ,is_load_checkpoint= load_checkpoint)
    # print("Multi-resolution done")