"""This code is refer to https://github.com/ZF4444/MMAL-Net"""

#coding=utf-8
import torch
import torch.nn as nn
import sys
from barbar import Bar
from config import input_size, proposalN, channels, model_name, model_path, batch_size, weight_path
from networks.model import MainNet
from torchvision.datasets import ImageFolder
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix, classification_report
from utils.myFunctions import accuracy
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from utils.myFunctions import *
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-data", "--dataset", required=True, help= "Choose dataset: IP102, D0 and Chengdata")
ap.add_argument("-dv", "--device", required= True, help= "Device type")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = vars(ap.parse_args())

if __name__ == '__main__':
    dataset_info = {
    'IP102' : 102,
    'D0' : 40
    }
    dataset_name = args['dataset']
    num_classes = dataset_info[dataset_name]
    if dataset_name == 'D0':
        dataset_name = 'unzip_D0'
    device = args['device']

    save_path = os.path.join(os.getcwd(), model_path, model_name + "_" + dataset_name)

    path_weight_dict = os.path.join(weight_path, model_name + '_' + dataset_name + '.pt')

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
    test_root_path = os.path.join(os.getcwd(), dataset_name, 'test')

    test_set = ImageFolder(root= test_root_path, transform= data_transforms['val'])
    testloader = DataLoader(test_set, batch_size= batch_size, shuffle= False,
                            num_workers= 8, pin_memory= True)

    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    model = model.to(device)

    model.load_state_dict(torch.load(path_weight_dict, map_location= 'cpu'))
    
    total = 0
    print('Testing')
    model.eval()
    topk=(1, 3, 5)
    y_test = None
    outputs = None
    predicted = None
    with torch.no_grad():
        for i, data in enumerate(Bar(testloader)):
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
            total += y.size(0)
                
            if y_test is None:
                y_test = y
            else:
                y_test = torch.cat((y_test, y))

            _, temp_predicted = torch.max(logits.data, 1)

            if predicted is None:
                predicted = temp_predicted
            else:
                predicted = torch.cat((predicted, temp_predicted))

        predicted = predicted.to('cpu')
        outputs = outputs.to('cpu')
        y_test = y_test.to('cpu')
        topKAccuracy = accuracy(outputs, y_test, topk= topk)

        path_folder = os.path.join(os.getcwd(), model_path, model_name  + '_' + dataset_name + '_torch')
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        with open(os.path.join(path_folder, ('Result {s}.txt').format(s = model_name)), 'w') as f:
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