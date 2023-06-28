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
    def __init__(self,input_dim,action_dim,max_action,hidden_sizes=[256,256],learning_rate= 0.0002,reparam_noise = 1e-6,name="actor",activation_fun= torch.nn.ReLU()):
        super(Actor,self).__init__(input_dim,hidden_sizes,output_size=action_dim,learning_rate=learning_rate,name=name)
        self.max_action = max_action
        self.reparam_noise = reparam_noise
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [activation_fun for l in self.layers]
        self.log_sigma = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.mu = torch.nn.Linear(layer_sizes[-1],action_dim)
    
    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        log_sigma = torch.clamp(log_sigma,-20,10)
        sigma = log_sigma.exp()
        return mu, sigma
        
    def sample_normal(self, state, reparameterize=True):
        mu,log_sigma = self.forward(state)
        probabilities = Normal(mu,log_sigma)
        if reparameterize:
            #sample with noise
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        action = torch.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)-torch.log(1-action.pow(2)+self.reparam_noise)
        #to scalar for loss
        if log_probs.shape != torch.Size([1]):
            log_probs = log_probs.sum(1,keepdim=True)
        return action, log_probs


