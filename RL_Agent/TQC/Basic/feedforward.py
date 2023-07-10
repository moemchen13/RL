import os
import torch
import torch.optim as optim
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(),
                 output_activation=None,device ='cpu',name="feedforward",folder="tmp"):
        super(Feedforward, self).__init__()
        self.device =device
        self.input_size = input_dim
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        self.output_activation = output_activation
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.checkpoint_file = os.path.join(folder,name)
        if self.device == 'cuda':
            self.cuda()

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
        
    def save_checkpoint(self):
        torch.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
