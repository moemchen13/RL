import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Distribution,Normal
from torch.nn.functional import relu, logsigmoid

class Actor(torch.nn.Module):
    def __init__(self,input_dim,action_dim,action_space=None,hidden_sizes=[256,256],device='cpu',
                 learning_rate= 0.0001,name="actor",
                 activation_fun= torch.nn.ReLU()):
        super(Actor,self).__init__()
        #NN
        self.device=device
        self.input_size = input_dim
        self.hidden_sizes = hidden_sizes
        layer_sizes = [self.input_size] + self.hidden_sizes + [2*action_dim]
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.log_sigma = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.mu = torch.nn.Linear(layer_sizes[-1],action_dim)

        self.learning_rate = learning_rate
        self.action_space = action_space
        self.min_log_std = torch.Tensor(-20,device=self.device)
        self.max_log_std = torch.Tensor(2,device=self.device)
        self.reparam_noise = torch.Tensor(1e-6,device=self.device)
        self.action_dim = action_dim

        if device =='cuda':
            self.cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self,state):
        
        for layer,activation_fun in zip(self.layers, self.activations):
            state = activation_fun(layer(state))
        
        mu,log_sigma = state.split([self.action_dim,self.action_dim],dim=1)
        log_sigma = torch.clamp(log_sigma,self.min_log_std,self.max_log_std)

        return mu,log_sigma
    

    def random_action(self):
        #gives random action for early exploration
        action = 2 * torch.rand(self.action_dim,device=self.device) - 1
        return action[None,:]


    def get_action(self,state):
        #deterministic gives best action
        mu,_ = self.forward(state)
        mu = torch.tanh(mu)
        return mu
    

    def get_action_and_log_probs(self, state,reparameterize=False):
        #stochastic best action for exploration near best action
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma)
        distribution = Normal(mu,sigma)
        if reparameterize:
            #Makes it differentiable (reparameterization trick)
            sample = distribution.rsample()
        else:
            sample = distribution.sample()

        action = torch.tanh(sample)
        log_prob = distribution.log_prob(sample)
        log_prob -= torch.log((1 - action.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum(axis=1,keepdim=True)
        
        return action, log_prob


