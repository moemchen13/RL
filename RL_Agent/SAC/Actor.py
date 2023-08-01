import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.set_num_threads(1)

class Actor(nn.Module):
    def __init__(self,input_dim,action_dim,hidden_sizes=[256,256],
                 learning_rate= 0.0001,name="actor",
                 activation_fun= torch.nn.ReLU(),device='cpu'):
        super().__init__()
        self.device = device
        layer_sizes = [input_dim] + hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])]).to(device=self.device)
        self.activations = [activation_fun for l in self.layers]
        self.log_sigma = torch.nn.Linear(layer_sizes[-1],action_dim).to(device=self.device)
        self.mu = torch.nn.Linear(layer_sizes[-1],action_dim).to(device=self.device)
        self.learning_rate = learning_rate
        self.min_log_std = torch.tensor(-20).to(self.device)
        self.max_log_std = torch.tensor(2).to(self.device)
        self.reparam_noise = torch.tensor(1e-6).to(self.device)
        self.action_dim = torch.tensor(action_dim).to(self.device)
        if self.device =='cuda':
            self.cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    
    def forward(self, state):
        print(f"dreckscuda {state.is_cuda}")
        for layer,activation_fun in zip(self.layers, self.activations):
            state = activation_fun(layer(state))
        
        mu = self.mu(state)
        log_sigma = self.log_sigma(state)
        log_sigma = torch.clamp(log_sigma,self.min_log_std,self.max_log_std)
        
        return mu, log_sigma

    def random_action(self):
        #random for early exploration
        action = 2 * torch.rand(self.action_dim,device=self.device) - 1
        return action

    def get_action(self,state):
        #deterministic gives best action
        mu,_ = self.forward(state)
        return mu
    

    def get_action_and_log_probs(self, state,reparameterize=False):
        #gives random action for late exploration
        print("action log_prob")
        mu, log_sigma = self.forward(state)
        print("forward fin")
        sigma = log_sigma.exp()
        distribution = Normal(mu,sigma)
        print("created dist")

        if reparameterize:
            #Makes it differentiable reparameterization trick
            sample = distribution.rsample()
        else:
            sample = distribution.sample()
        print("sampled")

        action = torch.tanh(sample)
        log_prob = distribution.log_prob(sample)
        log_prob -= torch.log((1 - action.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum(axis=1,keepdim=True)
        print("get_action_log_probs fin")
        return action, log_prob