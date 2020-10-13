import torch
import torch.nn.functional as F
import torch.nn as nn

class stackedGeneralization_Ensemble(nn.Module):
    """Stacking ensemble model, with Neural networks as the meta-model"""
    def __init__(self, n_models, num_classes= 102, n_hidden_node= 256):
        """Creates an stacking ensemble model instance.

        Args:
            n_models (integer): Number of models use for combination.
            num_classes (integer): Number of classes
            n_hidden_node (integer): Number of hidden node in the neural networks
        """
        super(stackedGeneralization_Ensemble, self).__init__()
        self.n_models = n_models
        self.num_classes = num_classes
        self.n_hidden_node = n_hidden_node
        input_lentent = self.n_models * self.num_classes
        
        self.fc = nn.Sequential(nn.Linear(input_lentent, self.n_hidden_node), 
                                nn.Tanh(), 
                                nn.BatchNorm1d(self.n_hidden_node),
                                nn.Linear(self.n_hidden_node, self.num_classes))


    def forward(self, x):
        ensamble_ouputs = self.fc(x)
        return ensamble_ouputs

class convGatingNetwork_Ensemble(nn.Module):
    """Convoluional Gating Network ensemble model"""
    def __init__(self, member_models, member_models_name, num_classes= 102, dropout= 0.5):
        """Creates a Convoluional Gating Network ensemble model instance.

        Args:
            member_models (nn.ModuleList): List of CNN experts.
            member_models_name (Iterable): List of CNN expert's names
            num_classes (integer): Number of classes
            dropout (float): Dropout rate
        """
        super(convGatingNetwork_Ensemble, self).__init__()
        self.member_models_name = member_models_name
        self.member_models = member_models

        for model in self.member_models:
            for param in model.parameters():
                param.requires_grad = False

        self.n_models = len(self.member_models)
        self.num_classes = num_classes

        self.conv_block = nn.Sequential(nn.Conv2d(3, 32, kernel_size= 7, stride= 2, padding= 3, bias=False), #112
                                        nn.ReLU(inplace= True),
                                        nn.BatchNorm2d(32),
                                        nn.MaxPool2d(kernel_size= 2, stride= 2), #56
                                        nn.Conv2d(32, 64, kernel_size= 5, stride= 1, padding= 2, bias=False), #56
                                        nn.ReLU(inplace= True),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d(kernel_size= 2, stride= 2), #28
                                        nn.Conv2d(64, 128, kernel_size= 3, stride= 2, padding= 1, bias=False), #14
                                        nn.ReLU(inplace= True),
                                        nn.BatchNorm2d(128),
                                        nn.MaxPool2d(kernel_size= 2, stride= 2) #7
                                        )

        self.fc = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(7 * 7 * 128, 256),
                                nn.ReLU(inplace= True),
                                nn.Dropout(dropout),
                                nn.Linear(256, self.n_models),
                                nn.Softmax(dim= 1))


    def forward(self, x1, x2):
        gating_out = self.conv_block(x1)
        gating_out = torch.flatten(gating_out, 1)
        gating_out = self.fc(gating_out) # (?, n,)
        gating_out = torch.unsqueeze(gating_out, 1)

        members_out = [] #(?, n, n_classes,)
        for model, name in zip(self.member_models, self.member_models_name):
            if name == "multi-reso":
                members_out.append(F.softmax(model(x1, x2), dim= 1)) 
            else:
                members_out.append(F.softmax(model(x1), dim= 1)) 
        
        members_out = torch.stack(members_out, 2)

        ensamble_ouputs = torch.sum(torch.mul(members_out, gating_out), dim= 2)

        return ensamble_ouputs