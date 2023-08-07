import os
import pickle

import gymnasium as gym
import memory as mem
import numpy as np
import torch
from Agent import UnsupportedSpace, agent
from DSAC_Actor import Actor
from DSAC_Critic import Critic
from gymnasium import spaces
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('Using device:', device)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class DSAC_Agent(agent):
    def __init__(self,observation_space,action_space,**userconfig):
        super().__init__(observation_space,action_space,**userconfig)

        self._config = {
           "start_steps":10000,
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "lr_actor": float(3e-4),
            "lr_critic": float(1e-3),
            "hidden_size_critic": [256,256],
            "hidden_size_actor": [256,256],
            "frequency_update_Q":1,
            "frequency_update_actor":1,
            "frequency_update_targets":1,
            "tau": 0.005,
            "autotuned_temperature":True,
            "temperature":0.1,
            "smoothing_trick":False, 
            "number_critics":2,
            "stochastic_actor":True,
            "adaptive_bounds":True,
            "bounds":True,
            "TD_Bound":10,
            "bound":True,
            "play_hockey":True,
        }
        self.device = device
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self.action_dim = action_space.shape[0]
        if self._config["play_hockey"]:
            self.action_dim = action_space.shape[0]//2
        else:
            self.action_dim = action_space.shape[0]
            print(f"normal actionsspace of {self.action_dim}")
        self.discount = self._config["discount"]
        self.tau = self._config["tau"]
        self.train_iter=0
        self.eval_mode = False
        self.start_steps = self._config["start_steps"]
        self.memory = mem.Memory(max_size=self._config["buffer_size"],state_dim=self._obs_dim,
                                 action_dim=self.action_dim,device=self.device)

        if self._config["autotuned_temperature"]:
            self.target_entropy = -torch.Tensor(self.action_dim).to(self.device)
            self.log_temperature = torch.ones(1,requires_grad=True,device=self.device)
            self.temperature_optimizer = torch.optim.Adam([self.log_temperature],lr=self._config["lr_critic"])
        else:
            self.log_temperature = torch.Tensor(self._config["temperature"].log()).to(self.device)

        self.actor = Actor(self._obs_dim,self.action_dim,action_space=action_space,
                            learning_rate=self._config["lr_actor"],device=self.device)
        self.critic = Critic(self._obs_dim,self.action_dim,self._config["lr_critic"],
                               device=self.device)
        self.target = Critic(self._obs_dim,self.action_dim,self._config["lr_critic"],
                            tau=self.tau,device=self.device)
        
        self.target.soft_update(self.critic,tau=1)
        
        if self._config["play_hockey"]:
            self.action_scale = 1
            self.action_bias = 0
        else:
            if action_space is not None:
                self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2).to(self.device)
                self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2).to(self.device)
            else:
                self.action_scale = torch.tensor(1.).to(self.device)
                self.action_bias = torch.tensor(0.).to(self.device)


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
    
    def act(self,state):
        state = torch.FloatTensor(state).to(self.device)[None,:]
        
        if self.eval_mode:
            action,_,_ = self.actor.get_action_and_log_probs(state,deterministic=True)
        else:
            if self.start_steps> self.memory.size:
                action = self.actor.random_action()
            else:    
                action, _, _ = self.actor.get_action_and_log_probs(state)
        action = self.rescale_action(action)
        return action.cpu().detach().numpy()
    

    def target_q(self,rew,done,q,q_std,target_q_next,log_prob_a_next):
        td = rew + (1-done)*self._config["discount"]*(target_q_next- self.log_temperature.exp().detach()*log_prob_a_next)
        if self._config["adaptive_bounds"]:
            target_max = q + 3*q_std
            target_min = q - 3*q_std
            td = torch.min(target_max,td)
            td = torch.max(target_min,td)
        difference = torch.clamp(td-q,-self._config["TD_Bound"],self._config["TD_Bound"])
        td_q_bound = difference + q
        return td.detach(),td_q_bound.detach()


    def update_critic(self,q_val,q_std,target_q,target_q_std):
        if self._config["bound"]:
            loss = (torch.pow(q_val-target_q,2) / (2*torch.pow(q_std,2)) 
                    + torch.pow(q_val.detach()-target_q_std,2)/ (2*torch.pow(q_std,2))
                    + torch.log(q_std)).mean()
        else:
            loss  = -Normal(q_val,q_std).log_prob(target_q).mean()
        
        self.critic.update(loss)
        return loss.item()


    def update_policy(self,q_val,log_prob_next):
        loss = (self.log_temperature.exp().detach()*log_prob_next-q_val).mean()
        self.actor.update(loss)
        return loss.item()


    def update_temperature(self,log_probs):
        #TODO: Check dimensions
        self.temperature_optimizer.zero_grad()
        temperature_loss  =  -(self.log_temperature.exp() * 
                               (log_probs+ self.target_entropy).detach()).mean()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        return temperature_loss.item()


    def train(self,iter_fit=32):
        q_losses = []
        policy_losses = []
        temperature_losses=[]
        self.train_iter +=1
        batch_size=self._config["batch_size"]
        smoothing_trick =self._config["smoothing_trick"]
        update_q= self._config["frequency_update_Q"]
        update_actor = self._config["frequency_update_actor"]
        update_target = self._config["frequency_update_targets"] 

        for i in range(iter_fit):
            if self.memory.size > batch_size:
                data = self.memory.sample(batch=batch_size)
                s0,a,rew,s1,done = data

                
                q_val,q_std,_ = self.critic.evaluate(s0,a)
                a_next,log_prob_next,_ = self.actor.evaluate(s1)
                
                _, _,target_q_next_sample = self.target.evaluate(s1,a_next,min=False)
                target_q_val = target_q_next_sample

                if i % update_q==0:
                    with torch.no_grad():
                        target_q_1,target_q_1_std = self.target_q(rew,done,q_val,q_std,target_q_val,log_prob_next)
                    q_loss = self.update_critic(q_val,q_std,target_q_1,target_q_1_std)
                    q_losses.append(q_loss)

                if i % update_actor==0:
                    a_next, log_prob_next,_ = self.actor.evaluate(s0)

                    temperature_loss = self.update_temperature(log_prob_next)
                    q_val,_,_ = self.critic.evaluate(s0,a_next)
                    pi_loss = self.update_policy(q_val,log_prob_next)
                    temperature_losses.append(temperature_loss)
                    policy_losses.append(pi_loss)

                if i % update_target == 0:
                    self.target.soft_update(self.critic,self.tau)
        
        return q_losses,policy_losses,temperature_losses
