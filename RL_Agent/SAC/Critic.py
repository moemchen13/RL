import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from Basic import feedforward as NN

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(1)

class Critic(torch.nn.Module):
    def __init__(self, input_dim, action_dim, learning_rate, hidden_sizes=[256, 256],loss='l2',tau=None,target=False):
        super(Critic, self).__init__()
        self.tau = tau
        self.network_number = 2
        self.input_dim = input_dim + action_dim
        self.learning_rate = learning_rate
        self.hidden_sizes = [hidden_sizes]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.networks = []
        if loss == 'l2':
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.SmoothL1Loss(reduction='mean')

        if target:
            for i in range(self.network_number):
                target_critic = NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        learning_rate=self.learning_rate,name=f'target_critic_{i}')
                self.networks.append(target_critic)
        else:
            for i in range(self.network_number):
                critic = NN.Feedforward(input_dim=self.input_dim,hidden_sizes = hidden_sizes,output_size=1,
                                        learning_rate=self.learning_rate,name=f'critic_{i}')
                self.networks.append(critic)
        

    def soft_update(self,CriticNetwork,tau=None):
        if tau is None:
            tau = self.tau
        if tau is None:
            raise ValueError("This is a no TargetNetwork tau not specified")
        
        critic_networks = CriticNetwork.networks

        for i in range(self.network_number):
            critic = critic_networks[i]
            target = self.networks[i]
            for target_param, critic_param in zip(target.parameters(),critic.parameters()):
                target_param.data.copy_((1.0-tau)*target_param.data+tau*critic_param.data)


    def get_network_states(self):
        state_dicts = []
        for network in self.networks:
            state_dicts.append(network.state_dict())
        return state_dicts


    def load_network_states(self,states):
        for i in range(self.network_number):
            network = self.networks[i]
            network.load_state_dict(states[i])


    def update_critics(self,state,action,target):
        self.optimizer.zero_grad()

        pred_Q = torch.zeros(1,requires_grad=True)
        for i in range(self.network_number):
            network = self.networks[i]
            single_q_prediction = network.forward(state,action)
            pred_Q += single_q_prediction

        Q_loss = self.loss(pred_Q,target)
        Q_loss.backward()
        self.optimizer.step()
        return Q_loss.item()


    def get_min_Q_value(self,state,action):
        min_Q = None
        for i in range(self.network_number):
            network = self.networks[i]
            q_val = network.forward(torch.cat([state,action],dim=1))
            if i==0:
                min_Q = q_val
            else:
                torch.min(min_Q,q_val)
        return min_Q
    

        


