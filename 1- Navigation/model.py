import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size,64)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()  
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU() 
        self.bn5 = nn.BatchNorm1d(64)
        self.drop5 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(64, action_size)
        self.softmax1 = nn.Softmax() 

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out = self.fc1(state)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.drop2(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.bn5(out)  
        out = self.drop5(out)
        out = self.fc6(out)
        return out
