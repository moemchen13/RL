import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class Critic(nn.Module):
    def __init__(self,input_dim,action_dim,learning_rate,hidden_sizes=256,
                tau=None,device='cpu'):
        super().__init__()
        self.device = device
        self.tau = tau
        self.input_dim = input_dim + action_dim
        self.learning_rate =learning_rate
        self.hidden_sizes = hidden_sizes
        self.log_std_max = 4 #change
        self.log_std_min = -0.1 #change
        self.denominator = max(abs(self.log_std_min),self.log_std_max)

        self.shared_network = nn.Sequential(nn.Linear(self.input_dim,self.hidden_sizes),nn.GELU(),
                                            nn.Linear(self.hidden_sizes,self.hidden_sizes),nn.GELU(),
                                            nn.Linear(self.hidden_sizes,hidden_sizes),nn.GELU()).to(self.device)
        self.mean_network = nn.Sequential(nn.Linear(self.hidden_sizes,self.hidden_sizes),nn.GELU(),
                                          nn.Linear(self.hidden_sizes,self.hidden_sizes),nn.GELU(),
                                          nn.Linear(self.hidden_sizes,1)).to(self.device)
        self.std_network = nn.Sequential(nn.Linear(self.hidden_sizes,self.hidden_sizes),nn.GELU(),
                                         nn.Linear(self.hidden_sizes,self.hidden_sizes),nn.GELU(),
                                         nn.Linear(self.hidden_sizes,1)).to(self.device)

        if device =='cuda':
            self.cuda()

        params = list(self.shared_network.parameters())+ list(self.std_network.parameters())+list(self.mean_network.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
    

    def get_network_states(self):
        return self.shared_network.state_dict(),self.mean_network.state_dict(),self.std_network.state_dict()


    def load_network_states(self,state):
        self.shared_network.load_state_dict(state[0])
        self.mean_network.load_state_dict(state[1])
        self.std_network.load_state_dict(state[2])


    def forward(self,state,action):
        input = torch.cat([state,action],dim=1)
        input = self.shared_network(input)
        mean = self.mean_network(input)
        log_std = self.std_network(input)
        log_std = torch.clamp_min(self.log_std_max*torch.tanh(log_std/self.denominator),0) + \
                  torch.clamp_max(-self.log_std_min * torch.tanh(log_std / self.denominator), 0)
        return mean,log_std


    def evaluate(self,state,action,min=False):
        mean,log_std = self.forward(state,action)
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape,device=self.device),torch.ones(std.shape,device=self.device))

        if min:
            z = normal.sample()
            z = torch.clamp(z,-2,2)
        else:
            z = -torch.abs(normal.sample())
        
        q_val = mean + torch.mul(z,std)
        return mean,std,q_val

    def update(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def evaluate(self,state,action,min=False):
        mean, log_q_std = self.forward(state,action)
        std = log_q_std.exp()
        distribution  = Normal(mean,std)
        if min:
            sample = -torch.abs(distribution.sample())
        else:
            sample = distribution.sample()
            sample = torch.clamp(sample,mean-2,mean+2)

        return mean,std,sample
    
    def soft_update(self, critic,tau=None):
        if self.tau is None:
            raise ValueError("This is a no TargetNetwork tau not specified")
        if tau is None:
            tau=self.tau
        
        for target_param, critic_param in zip(self.shared_network.parameters(),critic.shared_network.parameters()):
            target_param.data.copy_((1.0-tau)*target_param.data+tau*critic_param.data)

        for target_param, critic_param in zip(self.mean_network.parameters(),critic.mean_network.parameters()):
            target_param.data.copy_((1.0-tau)*target_param.data+tau*critic_param.data)
        
        for target_param, critic_param in zip(self.std_network.parameters(),critic.std_network.parameters()):
            target_param.data.copy_((1.0-tau)*target_param.data+tau*critic_param.data)
