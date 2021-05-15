"""This file use for implement ensemble method"""

from utils.myFunctions import initialize_model, model_predict, accuracy, finegGrained_pred
from config import input_size, proposalN, channels, model_name, model_path, weight_path
from networks.model import MainNet
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-data", "--dataset", required=True, help= "Choose dataset: IP102 or D0")
ap.add_argument("-dv", "--device", required= True, help= "Device type")

dataset_info = {
    'IP102' : 102,
    'D0' : 40,
}

args = vars(ap.parse_args())

if __name__ == "__main__":
    ###########################################################
    batch_size = 64
    dropout = 0.3
    dataset_name = args['dataset']
    device = args['device']

    n_classes = dataset_info[dataset_name]
    if dataset_name == 'D0':
      dataset_name = 'unzip_D0'
    model_name = 'soft_voting'
    exp_name = 'insect_recognition'

    path_weight_dict1 = os.path.join(os.getcwd(), 'pre-trained'
    , 'resnet' + '_' + dataset_name + '.pt')

    path_weight_dict2 = os.path.join(os.getcwd(), 'pre-trained'
    , 'residual-attention' + '_' + dataset_name + '.pt')

    path_weight_dict3 = os.path.join(os.getcwd(), 'pre-trained'
    , 'fpn' + '_' + dataset_name + '.pt')

    path_weight_dict4 = os.path.join(os.getcwd(), 'pre-trained'
    , 'Fine-grained' + '_' + dataset_name + '.pt')
    ############################################################

    model1, input_size = initialize_model('resnet', n_classes, use_pretrained= False, dropout= dropout)
    model1.load_state_dict(torch.load(path_weight_dict1, map_location= 'cpu'))
    model1.to(device)
    print('Model loaded succesful')

    model2, _ = initialize_model('residual-attention', n_classes, use_pretrained= False)
    model2.load_state_dict(torch.load(path_weight_dict2, map_location= 'cpu'))
    model2.to(device)
    print('Model loaded succesful')
    
    model3, _ = initialize_model('fpn', num_classes= n_classes, use_pretrained= False)
    model3.load_state_dict(torch.load(path_weight_dict3, map_location= 'cpu'))
    model3.to(device)
    print('Model loaded succesful')

    model4 =  MainNet(proposalN=proposalN, num_classes= n_classes, channels=channels)
    model4.load_state_dict(torch.load(path_weight_dict4, map_location= 'cpu'))
    model4.to(device)
    print('Model loaded succesful')

    test_root_path = os.path.join(os.getcwd(), dataset_name, 'test')

    mtf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size)])

    tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        
    test_set = ImageFolder(root= test_root_path, transform= tf)
    test_set = DataLoader(test_set, batch_size= batch_size, shuffle= False, 
                            num_workers= 8, pin_memory= True)

    pred_m1, label1 = model_predict(model1, test_set, return_labels= True, device= device)
    pred_m2 = model_predict(model2, test_set, device= device)
    pred_m3 = model_predict(model3, test_set, device= device)  

    tf = transforms.Compose([
            transforms.Resize(480),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_set = ImageFolder(root= test_root_path, transform= tf)
    testloader = DataLoader(test_set, batch_size= 6, shuffle= False,
                            num_workers= 8, pin_memory= True)

    pred_m4 = finegGrained_pred(model4, testloader, device= device)

    pred = (pred_m1 + pred_m2 + pred_m3 + pred_m4) / 4
    pred_t = np.argmax(pred, axis= 1)
    pred_t = pred_t.numpy()

    topk = (1, 3, 5)
    topKAccuracy = accuracy(pred, label1, topk= topk)
    
    path_save_res = os.path.join(os.getcwd(), exp_name, model_name + '_' + dataset_name)
    if not os.path.exists(path_save_res):
        os.makedirs(path_save_res)
    with open(os.path.join(path_save_res, 'result.txt'), 'w') as f:
        f.write(str(classification_report(pred_t, label1.numpy(), 
                                        digits= np.int64(np.ceil(np.log(pred_t.shape[0]))))))
        for i, k in enumerate(topk):
            f.write('Top %d : %f\n' %(k, topKAccuracy[i]))
        f.write('Geometric mean : %f\n' %(geometric_mean_score(label1.numpy(), pred_t)))
    for i, k in enumerate(topk):
        print('Top %d : %f' %(k, topKAccuracy[i]))
    print('Report : \n', classification_report(pred_t, label1, 
                                        digits= np.int64(np.ceil(np.log(pred.shape[0])))))
    print('Geometric mean : %f\n' %(geometric_mean_score(label1.numpy(), pred_t)))