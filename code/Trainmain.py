from myFunctions import *
from mul_scale import mulinput_train_model, evaluate_mulinput_model, mulinput_ImageFolder
# from data import *
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse
from distutils.util import strtobool
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-optim", "--optimizer", required=True, help= "Optimizer function : Only SGD or Adam")
ap.add_argument("-sch", "--scheduler", required=True, help= "Scheduler:\nnone\nsteplr\nexpdecay\nmyscheduler")
ap.add_argument("-l2", "--weight_decay", required= True, help= "L2 regularzation")
ap.add_argument("-do", "--dropout", required= True, help= "Dropout rate")
ap.add_argument("-predt", "--use_pretrained", required= True, help= "Use pretrained model's weight")
ap.add_argument("-mn", "--model_name", required= True, help= "Model name:\nresnet\nresidual-attention\n\
fpn\nmulti-reso")
ap.add_argument("-lr", "--learning_rate", required= True, help= "Inital learning rate")
ap.add_argument("-bz", "--batch_size", required= True, help= "Batch size")
ap.add_argument("-ep", "--epochs", required= True, help= "Number of Epochs")
ap.add_argument("-dv", "--device", required= True, help= "Device type")

ap.add_argument("-istr", "--use_pretrained", required= True, help= "Use pretrained model's weight")
ap.add_argument("-isev", "--use_pretrained", required= True, help= "Use pretrained model's weight")
ap.add_argument("-savem", "--use_pretrained", required= True, help= "Use pretrained model's weight")
ap.add_argument("-saveck", "--use_pretrained", required= True, help= "Use pretrained model's weight")
ap.add_argument("-predt", "--use_pretrained", required= True, help= "Use pretrained model's weight")


args = vars(ap.parse_args())

if __name__ == "__main__":
    ###########################################################
    device = args['device']
    n_classes = 102
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

    exp_name = 'insect recognition'
    is_train = True # ENABLE TRAIN
    is_eval = True # ENABLE EVAL
    save_model_dict = True # SAVE MODEL
    is_save_checkpoint = True # SAVE CHECKPOINT
    load_model = False # MUST HAVE MODEL FIRST
    load_checkpoint = False # MUST HAVE CHECKPOINT MODEL FIRST
    checkpoint = 'checkpoint_' + model_name + '.pt'
    ############################################################

    if is_train == False and load_checkpoint == True:
        raise Exception('Error')

    if load_model and load_checkpoint:
        raise Exception('Error Dont load 2 as one')

    if load_model:
        path_weight_dict = (model_name + '.pt')
    else:
        path_weight_dict = None

    is_mulreso = False
    if model_name == "multi-reso":
        is_mulreso = True

    model_hyper = {
        'batch_size' : batch_size,
        'num_epochs' : num_epochs,
        'init_lr' : init_lr,
        'weight_decay' : weight_decay,
        'dropout' : dropout,
        'model_name' : model_name,
        'exp_name' : exp_name
    }

    model_ft, input_size = initialize_model(model_name, n_classes, dropout= dropout, use_pretrained= use_pretrained)

    if path_weight_dict:
        model_ft.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))

    if is_mulreso:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(input_size)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size)
            ])
        }
        train_set = mulinput_ImageFolder(root= 'train', transform= data_transforms['train']) 
        valid_set = mulinput_ImageFolder(root= 'valid', transform= data_transforms['val'])

    else:
        data_transforms = {
            'train': transforms.Compose([
               transforms.Resize(256),
                transforms.CenterCrop(input_size),             
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
        train_set = ImageFolder(root= 'train', transform= data_transforms['train']) 
        valid_set = ImageFolder(root= 'valid', transform= data_transforms['val'])

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

        if is_mulreso: 
            model_ft, hist = mulinput_train_model(model_ft, datasets_dict, criterion, 
                                        optimizer_ft, 
                                        scheduler= scheduler_ft, num_epochs= num_epochs,
                                        checkpointFn= checkpoint
                                        , device= device, is_save_checkpoint= is_save_checkpoint
                                        ,is_load_checkpoint= load_checkpoint)
        else:
            
            model_ft, hist = train_model(model_ft, datasets_dict, criterion, optimizer_ft, 
                                        scheduler= scheduler_ft, num_epochs= num_epochs,
                                        is_inception= model_name == 'inception', checkpointFn= checkpoint
                                        , device= device, is_save_checkpoint= is_save_checkpoint
                                        ,is_load_checkpoint= load_checkpoint)

    if save_model_dict:
        torch.save(model_ft.state_dict(), model_name + '.pt')

    if is_eval:
        if is_mulreso:
            test_set = mulinput_ImageFolder(root= 'test', transform= data_transforms['val'])
            test_set = DataLoader(test_set, batch_size= batch_size, shuffle= False,
                        num_workers= 8, pin_memory= True)
                        
            evaluate_mulinput_model(model_ft, testloader= test_set, 
                    path_weight_dict= None, device= device, model_hyper= model_hyper)
        else:
            test_set = ImageFolder(root= 'test', transform= data_transforms['val'])
            test_set = DataLoader(test_set, batch_size= batch_size, shuffle= False,
                        num_workers= 8, pin_memory= True)
                        
            evaluate_model(model_ft, testloader= test_set, 
                    path_weight_dict= None, device= device, model_hyper= model_hyper)

    if is_train:
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