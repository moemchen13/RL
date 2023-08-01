import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Basic import feedforward as NN
from torch.distributions.normal import Normal

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.set_num_threads(1)

class Critic_Q(nn.Module):
    def __init__(self, input_dim, action_dim, learning_rate, hidden_sizes=[256, 256],
                 loss='l2',tau=0,target=False,network_number = 2,device='cpu'):
        super().__init__()
        self.device=device
        self.tau = tau
        self.network_number = network_number
        self.input_dim = input_dim + action_dim
        self.learning_rate = learning_rate
        self.networks = []
        

        if target:
            self.networks = [NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        name=f'target_critic_{i}',device=self.device) for i in range(self.network_number)]
        else:
            self.networks = [NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        name=f'critic_{i}',device=self.device)for i in range(self.network_number)]
        if loss == 'l2':
            self.losses = [nn.MSELoss() for i in range(self.network_number)]
        else:
            self.losses = [nn.SmoothL1Loss(reduction='mean') for i in range(self.network_number)]
        
        if device =='cuda':
            self.cuda()

        
        #parameters = []
        #for net in self.networks:
        #    parameters += list(net.parameters())
        #self.optimizers = torch.optim.Adam(parameters,lr=self.learning_rate)

        self.optimizers = [optim.Adam(self.networks[i].parameters(), lr=self.learning_rate)for i in range(self.network_number)]

        

    def soft_update(self,CriticNetwork,tau=None):
        if self.tau == 0:
            raise ValueError("This is a no TargetNetwork tau not specified")
        
        if tau == None:
            tau = self.tau
        
        for target, critic in zip(self.networks,CriticNetwork.networks):
            for target_param, critic_param in zip(target.parameters(),critic.parameters()):
                target_param.data.copy_((1.0-tau)*target_param.data+tau*critic_param.data)


    def get_network_states(self):
        return [network.state_dict() for network in self.networks]
        

    def load_network_states(self,states):
        for network,state in zip(self.networks,states):
            network.load_state_dict(state)


    def update_critics(self,state,action,target):
        Start = True
        for optimizer,loss,network in zip(self.optimizers,self.losses,self.networks):
            optimizer.zero_grad()
            q_pred = network.forward(torch.cat([state,action],dim=1))
            Q_loss = loss(q_pred,target)*1/self.network_number
            Q_loss.backward()
            optimizer.step()

            if Start:
                Loss = Q_loss.item()
                Start = False
            else:
                Loss += Q_loss.item()
        return Loss
    

    def update_critics_DR3(self,s0,a,s1,a_prime,target,beta=0.01):
        Loss = None
        x = torch.cat([s0,a],dim=1)
        x_prime = torch.cat([s1,a_prime],dim=1)
        for optimizer,loss,network in zip(self.optimizers,self.losses,self.networks):
            optimizer.zero_grad()
            q_pred = network.forward(x)
            regularizer = network.dot_prod_last_layer(x,x_prime).sum()
            Q_loss = (loss(q_pred,target)+regularizer*beta)*1/self.network_number
            Q_loss.backward()
            optimizer.step()

            if Loss is None:
                Loss = Q_loss.item()
            else:
                Loss += Q_loss.item()
        return Loss
    

    def get_min_Q_value(self,state,action):
        to_create_min_Q = True
        for network in self.networks:
            q_val = network.forward(torch.cat([state,action],dim=1))

            if to_create_min_Q:
                min_Q = q_val
                to_create_min_Q=False
            else:
                torch.min(min_Q,q_val)
        return min_Q