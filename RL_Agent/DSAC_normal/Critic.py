import torch
import torch.nn as nn
import torch.optim as optim
from Basic import feedforward as NN


class Critic(nn.Module):
    def __init__(self,input_dim,action_dim,learning_rate,hidden_sizes=[256,256],
                 loss='l2',tau=None,target=False,device='cpu'):
        super().__init__()
        self.device = device
        self.tau = tau
        self.input_dim = input_dim + action_dim
        self.learning_rate =learning_rate
        self.hidden_sizes = hidden_sizes
        self.networks = []

        if target:
            self.networks = [NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        learning_rate=self.learning_rate,name=f'target_critic_{i}',device=self.device) for i in range(self.network_number)]
        else:
            self.networks = [NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        learning_rate=self.learning_rate,name=f'critic_{i}',device=self.device)for i in range(self.network_number)]
        if loss == 'l2':
            self.losses = [nn.MSELoss() for i in range(self.network_number)]
        else:
            self.losses = [nn.SmoothL1Loss(reduction='mean') for i in range(self.network_number)]
        
        if device =='cuda':
            self.cuda()

        self.optimizers = [optim.Adam(self.networks[i].parameters(), lr=self.learning_rate)for i in range(self.network_number)]

