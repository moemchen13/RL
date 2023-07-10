import numpy as np
import torch
import torch.nn as nn
from Basic import feedforward as NN

class Critic_Quantiles(nn.Module):
    def __init__(self,input_dim,action_dim,number_quantiles,number_networks,
                 hidden_sizes=[512,512,512],device='cpu',lr=3e-4,tau=None,target=False):
        super(Critic_Quantiles,self).__init__()
        self.device = device
        self.tau = tau
        self.n_networks = number_networks
        self.n_quantiles = number_quantiles
        self.learning_rate = lr
        network_dim = input_dim+action_dim
        #Ask tutor need for self.add_module() needed for backpropagation?
        if target:
            self.networks = [NN.Feedforward(input_dim=network_dim,hidden_sizes=hidden_sizes,
                                            output_size=self.n_quantiles,device=self.device,
                                        name=f'target_{i}') for i in range(self.n_networks)]
        else:
            self.networks = [NN.Feedforward(input_dim=network_dim,hidden_sizes=hidden_sizes,
                                            output_size=self.n_quantiles,device=self.device,
                                        name=f'critic_{i}') for i in range(self.n_networks)]
            
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
