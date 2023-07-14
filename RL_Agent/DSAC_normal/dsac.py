import os
import pickle

import gymnasium as gym
import numpy as np
import torch
from Actor import Actor
from Basic import feedforward as NN
from Basic import memory as mem
from Basic.Agent import UnsupportedSpace, agent
from Critic import Critic
from gymnasium import spaces

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class DSAC_Agent(agent):
    def __init__(self,observation_space,action_space,**userconfig):
        super().__init__(self,observation_space,action_space,**userconfig)

        self._config = {
           "start_steps":10000,
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "lr_actor": float(3e-4),
            "lr_critic": float(1e-3),
            "hidden_size_critic": [128,128],
            "hidden_size_actor": [128,128],
            "frequency_update_Q":1,
            "frequency_update_actor":1,
            "frequency_update_targets":1,
            "tau": 0.005,
            "update_target_every":1,
            "autotuned_temperature":True,
            "temperature":0.1,
            "use_smooth_L1":False, 
            "number_critics":2,
        }
        self.device = device
        self.device = device
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self.action_dim = action_space.shape[0]
        self.discount = self._config["discount"]
        self.tau = self._config["tau"]
        self.train_iter=0
        self.eval_mode = False
        self.start_steps = self._config["start_steps"]
        self.memory = mem.Memory(max_size=self._config["buffer_size"],state_dim=self._obs_dim,action_dim=self.action_dim)

        if self._config["autotuned_temperature"]:
            self.target_entropy = -torch.Tensor(self.action_dim,device=self.device)
            self.log_temperature = torch.zeros(1,requires_grad=True,device=self.device)
            self.temperature_optimizer = torch.optim.Adam([self.log_temperature],lr=self._config["lr_critic"])
        else:
            self.log_temperature = torch.Tensor(self._config["temperature"].log(),device=self.device)

        self.actor = Actor(self._obs_dim,self.action_dim,action_space=action_space,hidden_sizes=self._config["hidden_size_actor"],
                            learning_rate=self._config["lr_actor"],device=self.device)
        self.critic = Critic(self._obs_dim,self.action_dim,self._config["lr_critic"],
                               hidden_sizes=self._config["hidden_size_critic"],device=self.device)

        self.target = Critic(self._obs_dim,self.action_dim,self._config["lr_critic"],
                            hidden_sizes=self._config["hidden_size_critic"],
                            tau=self.tau,target=True,device=self.device)
        
        self.target.soft_update(self.critic,tau=1)
        
        if action_space is not None:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2,device=self.device)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2,device=self.device)
        else:
            self.action_scale = torch.tensor(1.,device=self.device)
            self.action_bias = torch.tensor(0.,device=self.device)


    def store_transition(self, transition):
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

    def train(self,iter_fit=32):
        q_losses = []
        policy_losses = []
        temperature_losses=[]
        self.train_iter +=1

        for i in range(iter_fit):
            if self.memory.size > self._config['batch_size']:
                pass

        return q_losses,policy_losses,temperature_losses
