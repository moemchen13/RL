import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self,input_dim,action_dim,action_space=None,hidden_sizes=[256,256],
                 learning_rate= 1e-4,name="actor",activation_fun = torch.nn.ReLU(),
                 device='cpu'):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.action_dim = action_dim

        if self.device == 'cuda':
            self.cuda()
