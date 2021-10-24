"""
pytorch models for implementing DDPG
"""
import torch
from torch import nn
import torch.nn.functional as F


class ActorModel(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size_1=128, hidden_size_2=256):
        """
        :param state_size: the number of floats in a state
        :param action_size: the number of floats in an action
        :param seed: random seed
        :param hidden_size_1: the size of the first hidden layer
        :param hidden_size_2: the size of the second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, action_size)
        self.batchnorm_1 = nn.BatchNorm1d(hidden_size_1)

    def forward(self, state):
        x = F.relu(self.batchnorm_1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class CriticModel(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size_1=128, hidden_size_2=256):
        """
        :param state_size: the number of floats in a state
        :param action_size: the number of floats in an action
        :param seed: random seed
        :param hidden_size_1: the size of the first hidden layer
        :param hidden_size_2: the size of the second hidden layer (less the action size)
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1+action_size, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)
        self.batchnorm_1 = nn.BatchNorm1d(hidden_size_1)

    def forward(self, state, action):
        xs = F.relu(self.batchnorm_1(self.fc1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
