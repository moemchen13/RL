import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from Basic import feedforward as NN

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(1)

class Actor(NN.Feedforward):
    def __init__(self,input_dim,action_dim,action_space=None,hidden_sizes=[256,256],
                 learning_rate= 0.0001,name="actor",
                 activation_fun= torch.nn.ReLU()):
        super(Actor,self).__init__(input_dim,hidden_sizes,output_size=action_dim,learning_rate=learning_rate,name=name)
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [activation_fun for l in self.layers]
        self.log_sigma = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.mu = torch.nn.Linear(layer_sizes[-1],action_dim)
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.min_log_std = -20
        self.max_log_std = 2
        self.reparam_noise = 1e-6
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        if action_space is not None:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2)
        else:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)

    
    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)

        #alternative way to clamping log_sigma
        #log_std = torch.tanh(log_std)
        #log_std_min, log_std_max = self.log_std_bounds
        #log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)

        log_sigma = torch.clamp(log_sigma,self.min_log_std,self.max_log_std)
        
        return mu, log_sigma

    def random_action(self):
        action = 2 * torch.rand(self.action_dim) - 1
        action = action * self.action_scale + self.action_bias
        return action

    def get_action(self,state):
        #deterministic
        mu,_ = self.forward(state)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias
        return mu
    

    def get_action_and_log_probs(self, state,reparameterize=False):
        #stochastic
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        distribution = Normal(mu,sigma)

        if reparameterize:
            #Makes it differentiable reparameterization trick
            sample = distribution.rsample()
        else:
            sample = distribution.sample()

        y = torch.tanh(sample)
        
        action = y * self.action_scale + self.action_bias
        log_prob = distribution.log_prob(sample)
        #TODO undo if no effect
        #log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + self.reparam_noise)
        #if not only vector
        if self.action_dim>1:
            #if log_prob.dim()==1:
            #    log_prob = log_prob.unsqueeze(axis=0).sum(axis=1,keepdim=True)
            #else:
            log_prob = log_prob.sum(axis=1,keepdim=True)
        
        return action, log_prob