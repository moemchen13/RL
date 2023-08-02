import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size

        print(self.input_size, self.hidden_sizes[0], self.hidden_sizes[1], self.output_size)

        self.input_connector = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_sizes[0])
        )

        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_sizes[1],1)
        )

        self.advantage_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_sizes[1], self.output_size)
        )

    def forward(self, x):
        x = self.input_connector(x)
        values = self.value_layer(x)
        advantages = self.advantage_layer(x)
        q_values = values + (advantages - advantages.mean())
        return q_values
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()