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
    def __init__(self, gNet_ft, member_models, member_models_name, num_classes= 102, dropout= 0.5):
        """Creates a Convoluional Gating Network ensemble model instance.

        Args:
            member_models (nn.ModuleList): List of CNN experts.
            member_models_name (Iterable): List of CNN expert's names
            features_extractor (nn.Module): Extractor for gating network
            num_classes (integer): Number of classes
            dropout (float): Dropout rate
        """
        super(convGatingNetwork_Ensemble, self).__init__()
        self.gNet_ft = gNet_ft
        self.member_models_name = member_models_name
        self.member_models = member_models

        for model in self.member_models:
            for param in model.parameters():
                param.requires_grad = False

        self.n_models = len(self.member_models)
        self.num_classes = num_classes

        for param in self.gNet_ft.parameters():
            param.requires_grad = False

        num_fts = self.gNet_ft.fc[-1].in_features

        self.gNet_ft.fc = nn.Sequential(nn.Dropout(dropout),
                          nn.Linear(num_fts, self.n_models),
                          nn.Softmax(dim= 1))


    def forward(self, x1, x2):
        gating_out = self.gNet_ft(x1) # (?, n,)
        gating_out = torch.unsqueeze(gating_out, 1)

        members_out = [] #(?, n, n_classes,)
        for model, name in zip(self.member_models, self.member_models_name):
            if name == "multi-reso":
                members_out.append(model(x1, x2)) 
            else:
                members_out.append(model(x1)) 
        
        members_out = torch.stack(members_out, 2)

        ensamble_ouputs = torch.sum(torch.mul(members_out, gating_out), dim= 2)

        return ensamble_ouputs