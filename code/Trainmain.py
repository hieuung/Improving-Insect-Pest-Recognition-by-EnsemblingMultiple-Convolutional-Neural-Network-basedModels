"""This file is the main training and testing file, except for Fine-Grained model"""

from utils.myFunctions import train_model, evaluate_model, initialize_model, myScheduler
import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
import argparse
from distutils.util import strtobool
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-data", "--dataset", required=True, help= "Choose dataset: IP102 or D0")
ap.add_argument("-optim", "--optimizer", required=True, help= "Optimizer function : Only SGD or Adam")
ap.add_argument("-sch", "--scheduler", required=True, help= "Scheduler:\nnone\nsteplr\nexpdecay\nmyscheduler")
ap.add_argument("-l2", "--weight_decay", required= True, help= "L2 regularzation")
ap.add_argument("-do", "--dropout", required= True, help= "Dropout rate")
ap.add_argument("-predt", "--use_pretrained", required= True, help= "Use pretrained model's weight")
ap.add_argument("-mn", "--model_name", required= True, help= "Model name:\nresnet\nresidual-attention\n\
fpn")
ap.add_argument("-lr", "--learning_rate", required= True, help= "Inital learning rate")
ap.add_argument("-bz", "--batch_size", required= True, help= "Batch size")
ap.add_argument("-ep", "--epochs", required= True, help= "Number of Epochs")
ap.add_argument("-dv", "--device", required= True, help= "Device type")

ap.add_argument("-istra", "--istrain", required= False, help= "Train mode", default='True')
ap.add_argument("-iseva", "--iseval", required= False, help= "Eval mode", default='True')
ap.add_argument("-issavck", "--issavechkp", required= False, help= "Save checkpoint", default='True')
ap.add_argument("-issavmd", "--issavemodel", required= False, help= "Save model", default='True')
ap.add_argument("-isloadck", "--isloadchkp", required= False, help= "Load checkpoint", default='False')
ap.add_argument("-isloadmd", "--isloadmodel", required= False, help= "Load model", default='False')

args = vars(ap.parse_args())

dataset_info = {
    'IP102' : 102,
    'D0' : 40
}

if __name__ == "__main__":
    ###########################################################
    device = args['device']
    batch_size = int(args['batch_size'])
    num_epochs = int(args['epochs'])

    # My setting, Hyperparameters
    init_lr = float(args['learning_rate'])
    weight_decay = float(args['weight_decay'])
    dropout = float(args['dropout'])
    optimizer = args['optimizer']
    scheduler = args['scheduler']
    use_pretrained = strtobool(args['use_pretrained'])
    model_name = args['model_name']
    dataset_name = args['dataset']
    n_classes = dataset_info[dataset_name]

    exp_name = 'insect_recognition'
    is_train = strtobool(args['istrain']) # ENABLE TRAIN
    is_eval = strtobool(args['iseval']) # ENABLE EVAL
    save_model_dict = strtobool(args['issavemodel']) # SAVE MODEL
    is_save_checkpoint = strtobool(args['issavechkp']) # SAVE CHECKPOINT
    load_model = strtobool(args['isloadmodel']) # MUST HAVE MODEL FIRST
    load_checkpoint = strtobool(args['isloadchkp']) # MUST HAVE CHECKPOINT MODEL FIRST
    checkpoint = 'checkpoint_' + dataset_name + '_' + model_name + '.pt'
    ############################################################

    if is_train == False and load_checkpoint == True:
        raise Exception('Error, checkpoint can be load during training')

    if load_model and load_checkpoint:
        raise Exception('Error, conflict between checkpoint and model weight')

    if dataset_name == 'D0':
        dataset_name = 'unzip_D0'

    path_weight_dict = os.path.join(os.getcwd(), 'pre-trained', model_name + '_' + dataset_name + '.pt')

    model_hyper = {
        'batch_size' : batch_size,
        'num_epochs' : num_epochs,
        'init_lr' : init_lr,
        'weight_decay' : weight_decay,
        'dropout' : dropout,
        'model_name' : model_name,
        'exp_name' : exp_name,
        'dataset' : dataset_name
    }

    train_root_path = os.path.join(os.getcwd(), dataset_name, 'train')
    valid_root_path = os.path.join(os.getcwd(), dataset_name, 'valid')
    test_root_path = os.path.join(os.getcwd(), dataset_name, 'test')

    model_ft, input_size = initialize_model(model_name, n_classes, dropout= dropout, use_pretrained= use_pretrained)

    if load_model:
        model_ft.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))

    data_transforms = {
            'train': transforms.Compose([
               transforms.Resize(256),
                transforms.RandomCrop(input_size),             
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),               
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    train_set = ImageFolder(root= train_root_path, transform= data_transforms['train']) 
    valid_set = ImageFolder(root= valid_root_path, transform= data_transforms['val'])

    if is_train:
        train_set = DataLoader(train_set, batch_size= batch_size, shuffle= True, 
                            num_workers= 8, pin_memory=True)

        valid_set = DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
                                num_workers= 8, pin_memory= True)

        datasets_dict = {'train': train_set, 'val': valid_set}

        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()

        if optimizer.lower() == 'sgd':
            optimizer_ft = optim.SGD(params_to_update, lr= init_lr, momentum= 0.9, weight_decay= weight_decay)
        elif optimizer.lower() == 'adam':
            optimizer_ft = optim.Adam(params_to_update, lr= init_lr, betas= (0.9, 0.999),
                                eps= 1e-08, weight_decay= weight_decay)

        if scheduler.lower() == 'none':
            scheduler_ft = None
        elif scheduler.lower() == 'expdecay':
            scheduler_ft = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma= 0.96)
        elif scheduler.lower() == 'steplr':
            scheduler_ft = optim.lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma= 0.09999, last_epoch=-1)
        elif scheduler.lower() == 'myscheduler':
            scheduler_ft = myScheduler(optimizer_ft, gamma= 0.09999)

        criterion = nn.CrossEntropyLoss()

        model_ft, hist = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                        scheduler= scheduler_ft, num_epochs= num_epochs, checkpointFn= checkpoint
                                        , device= device, is_save_checkpoint= is_save_checkpoint
                                        ,is_load_checkpoint= load_checkpoint)

    if save_model_dict:
        torch.save(model_ft.state_dict(), path_weight_dict)

    if is_eval:
        test_set = ImageFolder(root= test_root_path, transform= data_transforms['val'])
        test_set = DataLoader(test_set, batch_size= batch_size, shuffle= False,
                            num_workers= 8, pin_memory= True)
                            
        evaluate_model(model_ft, testloader= test_set, 
                        path_weight_dict= None, device= device, model_hyper= model_hyper)

    if is_train:
        train_acc = [h.cpu().numpy() for h in hist['train_acc']]
        val_acc = [h.cpu().numpy() for h in hist['val_acc']]

        fig = plt.figure()
        path_folder = os.path.join(os.getcwd(), exp_name, model_name + '_' + dataset_name + '_torch')
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        plt.subplot(2, 1, 1)
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,len(train_acc) + 1), train_acc, label= "Train")
        plt.plot(range(1,len(val_acc) + 1), val_acc, label= "Val")
        plt.ylim((0,1.))
        plt.xticks(np.arange(1, len(train_acc) + 1, 1.0))
        plt.legend()

        train_loss = hist['train_loss']
        val_loss = hist['val_loss']

        plt.subplot(2, 1, 2)
        plt.title("Loss vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1,len(train_acc) + 1), train_loss, label= "Train")
        plt.plot(range(1,len(val_loss) + 1), val_loss, label= "Val")
        plt.xticks(np.arange(1, len(train_acc) + 1, 1.0))
        plt.legend()
        plt.savefig(os.path.join(path_folder, model_name + '.png'))
        plt.show()

        ran_eps = len(val_acc)

        with open(os.path.join(path_folder, (model_name + '.txt')), 'w') as f:
            f.write('Number epochs : %d\n' %(ran_eps))
            f.write('Learning rate : %f\n' %(init_lr))
            f.write('L2 regularization lambda: %f\n' %(weight_decay))
            for i in range(ran_eps):
                f.write('Epoch %d :' %(i + 1))
                f.write('Train acc : %f, Train loss : %f, Val acc : %f, Val loss : %f\n'
                        %(train_acc[i], train_loss[i], val_acc[i], val_loss[i]))