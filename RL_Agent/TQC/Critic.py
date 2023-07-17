import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = nn.functional.relu(fc(h))
        output = self.last_fc(h)
        return output


class Critic_Quantiles(nn.Module):
    def __init__(self,input_dim,action_dim,number_quantiles,number_networks,
                 hidden_sizes=[512,512,512],device='cpu',lr=3e-4,tau=None,target=False):
        super().__init__()
        self.device = device
        self.tau = tau
        self.n_networks = number_networks
        self.n_quantiles = number_quantiles
        self.learning_rate = lr
        network_dim = input_dim+action_dim
        #Ask tutor need for self.add_module() needed for backpropagation?
        
        self.networks = [MLP(input_size=network_dim,hidden_sizes=hidden_sizes,
                            output_size=self.n_quantiles) for i in range(self.n_networks)]
        
        parameters = []
        for net in self.networks:
            parameters += list(net.parameters())
        
        self.optimizer = torch.optim.Adam(parameters,lr=self.learning_rate)


    def forward(self,state,action):
        # batch,nets,quantiles
        input = torch.cat([state,action],dim=1)
        quantiles = torch.stack(tuple(net.forward(input) for net in self.networks),dim=1)
        return quantiles

    
    def soft_update(self,CriticNetwork,tau=None):
        if self.tau is None:
            raise ValueError("This is a no TargetNetwork tau not specified")
        if tau is None:
            tau=self.tau
        
        for target, critic in zip(self.networks,CriticNetwork.networks):
            for target_param, critic_param in zip(target.parameters(),critic.parameters()):
                target_param.data.copy_((1.0-tau)*target_param.data+tau*critic_param.data)


    def get_network_states(self):
        return [network.state_dict() for network in self.networks]
        

    def load_network_states(self,states):
        for network,state in zip(self.networks,states):
            network.load_state_dict(state)