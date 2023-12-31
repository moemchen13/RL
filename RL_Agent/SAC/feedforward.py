import os

import numpy as np
import torch
import torch.optim as optim


class Feedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_size, 
                 activation_fun=torch.nn.Tanh(),name="feedforward",folder="tmp",device='cpu'):
        super().__init__()
        self.device= device
        self.input_size = input_dim
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])]).to(self.device)
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size).to(self.device)
        self.checkpoint_file = os.path.join(folder,name)
        if self.device == 'cuda':
            self.cuda()
        

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        return self.readout(x)
    
    def dot_prod_last_layer(self,x,x_prime):
        for layer,activation_fun in zip(self.layers[:-1],self.activations[:-1]):
            x = activation_fun(layer(x))
            x_prime = activation_fun(layer(x_prime))
            #shape(x) = Batchsize,hidden_sizes[-1]
        #dot_product = (x*x_prime).sum(axis=1)
        dot_product = x.T @ x_prime
        return dot_product

    def save_checkpoint(self):
        torch.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
