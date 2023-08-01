import os
import pickle

import gymnasium as gym
import numpy as np
import torch
from Actor import Actor
from Agent import UnsupportedSpace, agent
from Basic import feedforward as NN
from Basic import memory as mem
from Critic import Critic_Q
from gymnasium import spaces

#device = torch.device('cpu')
#https://cloud.cs.uni-tuebingen.de/index.php/s/pm49B2xRpcNirry
#https://arxiv.org/pdf/2010.09163.pdf
#https://github.com/BY571/Soft-Actor-Critic-and-Extensions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class SAC_Agent(agent):
    def __init__(self,observation_space,action_space, **userconfig):
        
        super().__init__(observation_space,action_space,**userconfig)
        
        self._config = {
            "start_steps":1000, #10000
            "discount": 0.99,
            "buffer_size": int(1e7),
            "batch_size": 256,
            "lr_actor": float(3e-4),
            "lr_critic": float(1e-3),
            "lr_value": float(1e-3),
            "hidden_size_critic": [256,256],
            "hidden_size_actor": [256,256],
            "hidden_size_value": [256,256],
            "frequency_update_Q":1,
            "frequency_update_actor":1,
            "frequency_update_targets":1,
            "tau": 0.005,
            "update_target_every":1,
            "autotuned_temperature":True,
            "temperature":0.1,
            "network_numbers_critic":2,
            "use_smooth_L1":False,
            }
        
        self.device = device
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self.action_dim = action_space.shape[0]
        self.discount = self._config["discount"]
        self.tau = self._config["tau"]
        self.train_iter = 0
        self.eval_mode = False
        self.start_steps = self._config["start_steps"]

        self.memory = mem.Memory(max_size=self._config["buffer_size"],state_dim=self._obs_dim,action_dim=self.action_dim,device=self.device)
        
        if self._config["autotuned_temperature"]:
            self.target_entropy = -torch.Tensor(self.action_dim).to(self.device)
            self.log_temperature = torch.ones(1,requires_grad=True,device=self.device)
            self.temperature_optimizer = torch.optim.Adam([self.log_temperature],lr=self._config["lr_critic"])
        else:
            self.log_temperature = torch.Tensor(self._config["temperature"].log()).to(self.device)

        self.actor = Actor(self._obs_dim,self.action_dim,hidden_sizes=self._config["hidden_size_actor"],
                            learning_rate=self._config["lr_actor"],device=self.device)
        self.critic = Critic_Q(self._obs_dim,self.action_dim,self._config["lr_critic"],
                               hidden_sizes=self._config["hidden_size_critic"],
                               network_number=self._config["network_number_critic"],device=self.device)

        self.target = Critic_Q(self._obs_dim,self.action_dim,self._config["lr_critic"],
                            hidden_sizes=self._config["hidden_size_critic"],tau=self.tau,target=True,
                            network_number=self._config["network_number_critic"],device=self.device)
        
        self.target.soft_update(self.critic,tau=1)
        
        if action_space is not None:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low)/2).to(self.device)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low)/2).to(self.device)
        else:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)


    def store_transition(self,transition):
        self.memory.add_transition(transition)

    
    def get_networks_states(self):
        return (self.actor.state_dict(),self.critic.get_network_states(),self.target.get_network_states())
    

    def load_network_states(self,state):
        self.actor.load_state_dict(state[0])
        self.critic.load_network_states(state[1])
        self.target.load_network_states(state[2])
    

    def reset(self):
        raise NotImplementedError("U might wanna reset the policy network for faster learning as in DR3")


    def eval(self):
        self.eval_mode = True
        print("Agent now in evaluation Mode")
    

    def train(self):
        self.eval_mode = False
        print("Agent now in training mode")


    def rescale_action(self,action):
        return action[0]*self.action_scale + self.action_bias


    def act(self,state):
        state = torch.FloatTensor(state).to(self.device)[None,:]
        if self.eval_mode:
            action = self.actor.get_action(state)
        else:
            if self.start_steps> self.memory.size:
                action = self.actor.random_action()
            else:    
                print("not random action")
                action, _ = self.actor.get_action_and_log_probs(state)
        action = self.rescale_action(action)
        return action.cpu().detach().numpy()

    
    def get_Q_value(self,state,action):
        Q_val = self.critic.get_min_Q_value(state,action)
        return Q_val
    

    def get_target_Q_value(self,state,action):
        #same as get_Q_value but from target networks
        target_Q_val = self.target.get_min_Q_value(state,action)
        return target_Q_val

    
    def update_Q_functions(self,s0,action,done,rew,s1):
        print("update Q")
        with torch.no_grad():
            a_next , log_prob_next = self.actor.get_action_and_log_probs(s1,reparameterize=False)
            min_Q_next = self.get_target_Q_value(s1,a_next)
            #get V estimate
            target_value = min_Q_next - self.log_temperature.exp() * log_prob_next
            y = (rew + self.discount * (1 - done)*target_value)

        q_loss = self.critic.update_critics(state=s0,action=action,target=y)
        print("update Q finished")
        return q_loss      


    def update_policy(self,s0):
        print("update policy")
        action, log_prob = self.actor.get_action_and_log_probs(s0,reparameterize=True)
        actor_Q = self.get_Q_value(s0,action)
        actor_loss = (-actor_Q+self.log_temperature.exp().detach()*log_prob).mean(axis=0)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        print("update policy finished")
        return actor_loss.item(), log_prob
    

    def update_temperature(self,log_probs):
        print("update temp")
        self.temperature_optimizer.zero_grad()
        temperature_loss  =  -(self.log_temperature.exp() * 
                               (log_probs+ self.target_entropy).detach()).mean()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        print("update temp finished")
        return temperature_loss.item()


    def train(self,iter_fit=32):
        q_losses = []
        policy_losses = []
        temperature_losses=[]

        self.train_iter +=1
        
        for i in range(iter_fit):
            #print(i)
            if self.memory.size > self._config["batch_size"]:
                #Sample Batches from the replay Buffer
                data=self.memory.sample(batch=self._config["batch_size"])
                s0,a,rew,s1,done=data
                
                ######Start SAC train loop#######
                #updateQ
                if i % self._config["frequency_update_Q"] == 0:
                    q_loss = self.update_Q_functions(s0,a,done,rew,s1)
                    q_losses.append(q_loss)

                #update policy
                if i % self._config["frequency_update_actor"] == 0:
                    actor_loss,log_prob = self.update_policy(s0)
                    policy_losses.append(actor_loss)
                    
                #Update temperature
                if self._config["autotuned_temperature"]:
                    temperature_loss = self.update_temperature(log_prob)
                else:
                    temperature_loss = torch.tensor(0.)
                temperature_losses.append(temperature_loss)

                #Update targets networks
                if i % self._config["frequency_update_targets"] == 0:
                    self.target.soft_update(self.critic)
        
        return q_losses,policy_losses,temperature_losses

