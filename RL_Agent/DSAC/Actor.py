import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self,input_dim,action_dim,action_space=None,hidden_sizes=256,
                 learning_rate= 1e-4,name="actor",act_fun = torch.nn.GELU(),
                 device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_sizes=hidden_sizes
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.action_dim = action_dim
        self.log_std_max = 1
        self.log_std_min = -5
        self.denominator = max(abs(self.log_std_min), self.log_std_max)
        self.reparam_noise = 1e-4
        self.action_range = torch.tensor(2) #might wanna check

        self.shared_network = nn.Sequential(nn.Linear(self.input_dim,self.hidden_sizes),act_fun,
                                            nn.Linear(self.hidden_sizes,self.hidden_sizes),act_fun,
                                            nn.Linear(self.hidden_sizes,hidden_sizes),act_fun).to(self.device)
        self.mean_network = nn.Sequential(nn.Linear(self.hidden_sizes,self.hidden_sizes),act_fun,
                                          nn.Linear(self.hidden_sizes,self.hidden_sizes),act_fun,
                                          nn.Linear(self.hidden_sizes,self.action_dim)).to(self.device)
        self.std_network = nn.Sequential(nn.Linear(self.hidden_sizes,self.hidden_sizes),act_fun,
                                         nn.Linear(self.hidden_sizes,self.hidden_sizes),act_fun,
                                         nn.Linear(self.hidden_sizes,self.action_dim)).to(self.device)

        if self.device == 'cuda':
            self.cuda()
        params = list(self.shared_network.parameters())+ list(self.std_network.parameters())+list(self.mean_network.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)


    def get_network_states(self):
        return self.shared_network.state_dict(),self.mean_network.state_dict(),self.std_network.state_dict()


    def load_network_states(self,state):
        self.shared_network.load_state_dict(state[0])
        self.mean_network.load_state_dict(state[1])
        self.std_network.load_state_dict(state[2])


    def forward(self,state):
        state = self.shared_network(state)
        mean = self.mean_network(state)
        log_std = self.std_network(state)
        log_std = torch.clamp_min(self.log_std_max*torch.tanh(log_std/self.denominator),0) + \
                  torch.clamp_max(-self.log_std_min * torch.tanh(log_std / self.denominator), 0)
        return mean,log_std


    def update(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self,state):
        mean,log_std = self.forward(state)
        distribution = Normal(torch.zeros(mean.shape,device= self.device),torch.ones(log_std.shape,device= self.device))
        z = distribution.sample()
        z = torch.clamp(z,-3,3)
        std = log_std.exp()
        action_0 = mean + torch.mul(z,std)
        action_norm = torch.tanh(action_0)
        action = torch.mul(self.action_range.to(self.device), action_norm)
        log_prob = Normal(mean, std).log_prob(action_0) \
                    -torch.log(1. - action_norm.pow(2) +self.reparam_noise) \
                    - torch.log(self.action_range.to(self.device))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob , std.detach()
         
    
    def get_action_and_log_probs(self,state,deterministic=False):
        mean,log_std = self.forward(state)
        
        std = log_std.exp()
        distribution = Normal(torch.zeros(mean.shape),torch.ones(std.shape))
        z = distribution.sample()

        std = log_std.exp()
        action_0 = mean + torch.mul(z, std)
        action_norm = torch.tanh(action_0)
        action = torch.mul(self.action_range, action_norm)
        log_prob = Normal(mean, std).log_prob(action_0)-torch.log(torch.ones(action_norm.shape)
                    - action_norm.pow(2) + self.reparam_noise) - torch.log(self.action_range)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        action_mean = torch.mul(self.action_range, torch.tanh(mean))
        action = action_mean.detach() if deterministic else action.detach()
        return action, log_prob.detach(), std.detach()


    def random_action(self):
        action = 2 * torch.rand(self.action_dim,device=self.device) - 1
        return action[None,:]

