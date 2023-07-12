import torch


class Feedforward(torch.nn.Module):
    def __init__(self,input_dim,hidden_sizes,output_size,activation_f = torch.nn.Tanh()):
        super().__init__()

        layer_sizes = [input_dim] + hidden_sizes + [output_size]
        
        self.model = torch.nn.ModuleList([torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [activation_f for layer in layer_sizes]

    def forward(self,x):
        for i,a in zip(self.model,self.activations):
            x = a(i(x))
        return x

            
    def save_checkpoint(self):
        torch.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


