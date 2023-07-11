import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(1)

class Actor(nn.Module):
    def __init__(self,input_dim,action_dim,action_space=None,hidden_sizes=[256,256],
                 learning_rate= 0.0001,name="actor",
                 activation_fun= torch.nn.ReLU(),device='cpu'):
        super(Actor,self).__init__()
        self.device=device
        layer_sizes = [input_dim] + hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [activation_fun for l in self.layers]
        self.log_sigma = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.mu = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.min_log_std = torch.tensor(-20).to(self.device)
        self.max_log_std = torch.tensor(2).to(self.device)
        self.reparam_noise = torch.tensor(1e-6).to(self.device)
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if device =='cuda':
            self.cuda()


    
    def forward(self, state):
        for layer,activation_fun in zip(self.layers, self.activations):
            state = activation_fun(layer(state))
        
        mu = self.mu(state)
        log_sigma = self.log_sigma(state)
        log_sigma = torch.clamp(log_sigma,self.min_log_std,self.max_log_std)
        
        return mu, log_sigma

    def random_action(self):
        #random for early exploration
        action = 2 * torch.rand(self.action_dim) - 1
        return action

    def get_action(self,state):
        #deterministic gives best action
        mu,_ = self.forward(state)
        return mu
    

    def get_action_and_log_probs(self, state,reparameterize=False):
        #gives random action for late exploration
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        distribution = Normal(mu,sigma)

        if reparameterize:
            #Makes it differentiable reparameterization trick
            sample = distribution.rsample()
        else:
            sample = distribution.sample()

        action = torch.tanh(sample)
        log_prob = distribution.log_prob(sample)
        log_prob -= torch.log((1 - action.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum(axis=1,keepdim=True)
        
        return action, log_prob