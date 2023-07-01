import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from Basic import feedforward as NN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class Actor(NN.Feedforward):
    def __init__(self,input_dim,action_dim,max_action,hidden_sizes=[256,256],
                 learning_rate= 0.0002,name="actor",
                 activation_fun= torch.nn.ReLU(),min_log_std=-20,max_log_std=2):
        super(Actor,self).__init__(input_dim,hidden_sizes,output_size=action_dim,learning_rate=learning_rate,name=name)
        self.max_action = max_action
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [activation_fun for l in self.layers]
        self.log_sigma = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.mu = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    
    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        log_sigma = torch.clamp(log_sigma,self.min_log_std,self.max_log_std)
        return mu, log_sigma


    def get_action(self,state):
        mu,log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        action = self.sample_action(mu,sigma)
        action *= self.max_action
        return action


    def sample_action(self,mu,sigma,reparametrize=False):
        if reparametrize:
            sample = Normal(mu,sigma).rsample()
        else:
            sample = Normal(mu,sigma).sample()
        action = torch.tanh(sample).detach()
        return action
    

    def get_log_probs(self,mu,sigma, action):
        sample_dist = Normal(mu,sigma)
        log_probs = sample_dist.log_prob(action)
        return log_probs


    def get_action_and_log_probs(self, state, reparameterize=False):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        action = self.sample_action(mu,sigma,reparameterize)
        log_probs = self.get_log_probs(mu,sigma,action)
        action *= self.max_action
        return action, log_probs
    
    def get_dist(self,state):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        return Normal(mu,sigma)


