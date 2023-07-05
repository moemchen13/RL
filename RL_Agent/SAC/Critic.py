import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from Basic import feedforward as NN

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(1)

class Critic_Q(nn.Module):
    def __init__(self, input_dim, action_dim, learning_rate, hidden_sizes=[256, 256],loss='l2',tau=None,target=False):
        super(Critic_Q, self).__init__()
        self.tau = tau
        self.network_number = 2
        self.input_dim = input_dim + action_dim
        self.learning_rate = learning_rate
        self.hidden_sizes = [hidden_sizes]
        self.networks = []
        

        if target:
            self.networks = [NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        learning_rate=self.learning_rate,name=f'target_critic_{i}') for i in range(self.network_number)]
        else:
            self.networks = [NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        learning_rate=self.learning_rate,name=f'critic_{i}')for i in range(self.network_number)]
        if loss == 'l2':
            self.losses = [nn.MSELoss() for i in range(self.network_number)]
        else:
            self.losses = [nn.SmoothL1Loss(reduction='mean') for i in range(self.network_number)]

        self.optimizers = [optim.Adam(self.networks[i].parameters(), lr=self.learning_rate)for i in range(self.network_number)]

        

    def soft_update(self,CriticNetwork,tau=None):
        if tau is None:
            tau = self.tau
        if tau is None:
            raise ValueError("This is a no TargetNetwork tau not specified")
        
        for target, critic in zip(self.networks,CriticNetwork.networks):
            for target_param, critic_param in zip(target.parameters(),critic.parameters()):
                target_param.data.copy_((1.0-tau)*target_param.data+tau*critic_param.data)


    def get_network_states(self):
        return [network.state_dict() for network in self.networks]
        

    def load_network_states(self,states):
        for network,state in zip(self.networks,states):
            network.load_state_dict(state)


    def update_critics(self,state,action,target):
        Loss = None
        for optimizer,loss,network in zip(self.optimizers,self.losses,self.networks):
            optimizer.zero_grad()
            q_pred = network.forward(torch.cat([state,action],dim=1))
            Q_loss = loss(q_pred,target)*1/self.network_number
            Q_loss.backward()
            optimizer.step()

            if Loss is None:
                Loss = Q_loss.item()
            else:
                Loss += Q_loss.item()
        return Loss


    def get_min_Q_value(self,state,action):
        min_Q = None
        for network in self.networks:
            q_val = network.forward(torch.cat([state,action],dim=1))

            if min_Q is None:
                min_Q = q_val
            else:
                torch.min(min_Q,q_val)
        return min_Q
    

        


