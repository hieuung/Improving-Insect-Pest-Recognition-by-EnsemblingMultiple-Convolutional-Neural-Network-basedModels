from ensemble import *
from mul_scale import multi_scale_Reso, mulinput_train_model, evaluate_mulinput_model, mulinput_ImageFolder
from myFunctions import initialize_model
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    ###########################################################
    device = 'cuda'
    n_classes = 102
    batch_size = 64
    num_epochs = 100

    # # My setting
    # init_lr = 1e-4
    init_lr = 1e-2
    weight_decay = 0.00000
    dropout = 0.5
    # optimizer = 'Adam'
    optimizer = 'SGD'
    scheduler = 'expDecay'

    # model_name = 'ensamble_stacking'
    model_name = 'ensamble_gatingnet'

    exp_name = 'insect_recognition'

    path_weight_dict1 = os.path.join(os.getcwd(), exp_name, 
    'resnet_torch', 'resnetadam_noAug.pt')

    path_weight_dict2 = os.path.join(os.getcwd(), exp_name, 
    'residual-attention_torch', 'res-att_new1.pt')

    path_weight_dict3 = os.path.join(os.getcwd(), exp_name, 
    'multi_scale_model_torch', 'res_fpn.pt')

    path_weight_dict4 = os.path.join(os.getcwd(), exp_name, 
    'multi_scale_model_torch', 'multi_resolution2.pt')

    is_train = True # ENABLE TRAIN
    is_eval = True # ENABLE EVAL
    save_model_dict = False # SAVE MODEL
    is_save_checkpoint = False # SAVE CHECKPOINT
    load_model = False # MUST HAVE MODEL FIRST
    is_plot_export_train_log = False # SAVE TRAIN LOG
    load_checkpoint = False # MUST HAVE CHECKPOINT MODEL FIRST
    checkpoint = 'checkpoint_' + model_name + '.pt'
    ############################################################

    if is_train == False and load_checkpoint == True:
        raise Exception('Error')

    if is_train == False and is_plot_export_train_log == True:
        raise Exception('Error')

    if load_model and load_checkpoint:
        raise Exception('Error Don\'t load 2 as one')

    if load_model:
        path_weight_dict = (model_name + '.pt')
    else:
        path_weight_dict = None

    model_hyper = {
        'batch_size' : batch_size,
        'num_epochs' : num_epochs,
        'init_lr' : init_lr,
        'dropout' : dropout,
        'weight_decay' : weight_decay,
        'model_name' : model_name,
        'exp_name' : exp_name
    }

    model1, input_size = initialize_model('resnet', n_classes, use_pretrained= False, dropout= dropout)
    model1.load_state_dict(torch.load(path_weight_dict1, map_location= 'cpu'))
    model1.to(device)
    print('Model loaded succesful')

    model2, _ = initialize_model('residual-attention', n_classes, use_pretrained= False, dropout= dropout)
    model2.load_state_dict(torch.load(path_weight_dict2, map_location= 'cpu'))
    model2.to(device)
    print('Model loaded succesful')
    
    model3, _ = initialize_model('fpn', num_classes= n_classes, pretrained= False)
    model3.load_state_dict(torch.load(path_weight_dict3, map_location= 'cpu'))
    model3.to(device)
    print('Model loaded succesful')

    model4, _ = initialize_model('multi-reso', n_classes, dropout= dropout, use_pretrained= False)
    model4.load_state_dict(torch.load(path_weight_dict4, map_location= 'cpu'))
    model4.to(device)
    print('Model loaded succesful')
    
    sub_models = nn.ModuleList([model1, model2, model3, model4])
    sub_models_name = ['Resnet50', 'Attention_res', 'Resnet_fpn', 'multi-reso']

    model_ft = convGatingNetwork_Ensemble(sub_models, sub_models_name, num_classes= n_classes, dropout= dropout)

    if path_weight_dict:
        model_ft.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size)                
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size)              
        ]),
    }

    train_set = mulinput_ImageFolder(root= 'train', transform= data_transforms['train'])
    train_set = DataLoader(train_set, batch_size= batch_size, shuffle= True, 
                            num_workers= 8, pin_memory=True)

    valid_set = mulinput_ImageFolder(root= 'valid', transform= data_transforms['val'])
    valid_set = DataLoader(valid_set, batch_size= batch_size, shuffle= False, 
                            num_workers= 8, pin_memory= True)
    datasets_dict = {'train': train_set, 'val': valid_set}

    if is_train:
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

        model_ft, hist = mulinput_train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                    scheduler= scheduler_ft, num_epochs= num_epochs,
                                     checkpointFn= checkpoint
                                    , device= device, is_save_checkpoint= is_save_checkpoint
                                    ,is_load_checkpoint= load_checkpoint)

    if save_model_dict:
        torch.save(model_ft.state_dict(), model_name + '.pt')


    if is_eval:
        test_set = mulinput_ImageFolder(root= 'test', transform= data_transforms['val'])
        test_set = DataLoader(test_set, batch_size= batch_size, shuffle= False,
                        num_workers= 8, pin_memory= True)
                        
        evaluate_mulinput_model(model_ft, testloader= test_set, 
                    path_weight_dict= None, device= device, model_hyper= model_hyper)

    if is_plot_export_train_log:
        train_acc = [h.cpu().numpy() for h in hist['train_acc']]
        val_acc = [h.cpu().numpy() for h in hist['val_acc']]

        fig = plt.figure(figsize= (20, 10))
        path_folder = os.path.join(os.getcwd(), exp_name, model_name + '_torch')
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
        plt.savefig(model_name + '.png')
        plt.show()

        with open(os.path.join(path_folder, (model_name + '.txt')), 'w') as f:
            f.write('Number epochs : %d\n' %(num_epochs))
            f.write('Learning rate : %f\n' %(init_lr))
            f.write('L2 regularization lambda: %f\n' %(weight_decay))
            for i in range(num_epochs):
                f.write('Epoch %d :' %(i + 1))
                f.write('Train acc : %f, Train loss : %f, Val acc : %f, Val loss : %f\n'
                        %(train_acc[i], train_loss[i], val_acc[i], val_loss[i]))
