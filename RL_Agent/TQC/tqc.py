import os
import torch
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import pickle
from Basic import feedforward as NN
from Basic import memory as mem
from Basic.Agent import agent,UnsupportedSpace
from Actor import Actor
from Critic import Critic_Quantiles


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class TQC_Agent(agent):
    def __init__(self,observation_space,action_space,device='cpu', **userconfig):
        
        super().__init__(observation_space,action_space,**userconfig)
        
        self._config = {
            "start_steps":1000,
            "discount": 0.99,
            "buffer_size": int(1e7),
            "batch_size": 128,
            "lr_actor": float(3e-4),
            "lr_critic": float(3e-4),
            "hidden_size_critic": [512,512,512],
            "hidden_size_actor": [256,256],
            "frequency_update_Q":1,
            "frequency_update_actor":1,
            "frequency_update_targets":1,
            "tau": 0.005,
            "reward_scale":2,
            "update_target_every":1,
            "use_smooth_L1":False,
            "number_critics":2,
            "number_quantiles":25,
            "drop_top_quantiles": 2,
            "autotuned_temperature":True,
            "temperature":0.01,
            }
        self.device = device
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self.action_dim = action_space.shape[0]
        self.memory = mem.Memory(self._obs_dim,self.action_dim,max_size=self._config["buffer_size"])
        self.discount = self._config["discount"]
        self.tau = self._config["tau"]
        self.train_iter=0
        self.eval_mode = False
        self.start_steps = self._config["start_steps"]
        
        self.total_quantiles = self._config["number_quantiles"]*self._config["number_critics"]
        self.total_quantiles_to_drop = self._config["drop_top_quantiles"]*self._config["number_critics"]

        self.actor = Actor(self._obs_dim,self.action_dim,action_space=action_space,hidden_sizes=self._config["hidden_size_actor"],
                           device=self.device,learning_rate=self._config["lr_actor"])
        
        self.critic = Critic_Quantiles(self._obs_dim,self.action_dim,self._config["number_quantiles"],
                                       self._config["number_critics"],hidden_sizes=self._config["hidden_size_critic"],
                                       device=self.device,lr=self._config["lr_critic"])
        self.target = Critic_Quantiles(self._obs_dim,self.action_dim,self._config["number_quantiles"],
                                       self._config["number_critics"],hidden_sizes=self._config["hidden_size_critic"],
                                       device=self.device,lr=self._config["lr_critic"],tau=self.tau,target=True)
        
        self.target_entropy = -torch.Tensor(self.action_dim,device=self.device)
        self.log_temperature = torch.Tensor(np.log(self._config["temperature"]),requires_grad=True,device=self.device)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature],lr=self._config["lr_critic"])
        self.learning_iterations = torch.zeros(1,device=self.device)
        self.target.soft_update(self.critic,tau=1)

        
        if action_space is not None:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2,device=self.device)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2,device=self.device)
        else:
            self.action_scale = torch.tensor(1.,device=self.device)
            self.action_bias = torch.tensor(0.,device=self.device)


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


    def scale_action(self,action):
        return action[0]*self.action_scale + self.action_bias

    def act(self,state):
        state = torch.FloatTensor(state).to(self.device)[None,:]
        
        if self.eval_mode:
            action = self.actor.get_action(state)
        else:
            if self.start_steps> self.memory.size:
                action = self.actor.random_action()
            else:    
                action, _ = self.actor.get_action_and_log_probs(state)
        action = self.scale_action(action)
        return action.cpu().detach().numpy()

    
    def calculate_critic_loss(self,s0,action,rew,done,s1):
        
        with torch.no_grad():
            a_next , log_prob_next = self.actor.get_action_and_log_probs(s1,reparameterize=False)
            #get quantiles from net dimensions: batch,networks,quantiles
            next_z = self.target.forward(s1,a_next)
            next_z,_ = torch.sort(next_z.reshape(self._config["batch_size"],-1))
            z_dropped = next_z[:,:self.total_quantiles-self.total_quantiles_to_drop]
            #get V estimate
            target_value =  z_dropped - self.log_temperature.exp() * log_prob_next
            # this is the td
            y = (rew + self.discount * (1 - done)*target_value)

        z = self.critic.forward(s0,action)
        q_loss = quantile_huber_loss(z,y)
        return q_loss


    def calculate_policy_and_temperature_loss(self,s0):
        
        action, log_prob = self.actor.get_action_and_log_probs(s0,reparameterize=True)
        actor_Q = self.critic.forward(s0,action)

        #dimensions:  batch,networks,quantiles
        policy_loss = (self.log_temperature.exp().detach()*
                       log_prob-actor_Q.mean(axis=2).mean(axis=1,keepdim=True)).mean()
        #policy_loss = (self.log_temperature.exp().detach()*log_prob-actor_Q.clone().mean(axis=2).mean(axis=1,keepdim=True)).mean()
        if self._config["autotuned_temperature"]:
            temperature_loss = self.calculate_temperature_loss(log_prob)
        else:
            temperature_loss = torch.tensor.zeros(1,device=self.device)

        return policy_loss, temperature_loss
    

    def calculate_temperature_loss(self,log_probs):
        #temperature_loss  =  (-self.log_temperature * (log_probs+ self.target_entropy).detach()).mean()
        temperature_loss  =  -(self.log_temperature * (log_probs+ self.target_entropy).detach()).mean()
        return temperature_loss


    def update_q_nets(self,q_loss):
        #update critics
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        return q_loss.item()
    

    def update_actor_net(self,policy_loss):        
        #update policy
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()
        return policy_loss.item()


    def update_temperature(self,temperature_loss):
        #update temperture
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        return temperature_loss.item()


    def train(self,iter_fit=32):

        q_losses = []
        policy_losses = []
        temperature_losses=[]

        for i in range(iter_fit):
            if self.memory.size > self._config["batch_size"]:
                #Sample Batches from the replay Buffer
                data=self.memory.sample(batch=self._config["batch_size"])
                s0,a,rew,s1,done=data

                ######Start SAC train loop#######

                #Calculate losses
                #update policy and Update temperature
                if self.train_iter % self._config["frequency_update_actor"] == 0:
                    actor_loss,temperature_loss = self.calculate_policy_and_temperature_loss(s0)
                    actor_loss = self.update_actor_net(actor_loss)
                    policy_losses.append(actor_loss)
                    temperature_loss = self.update_temperature(temperature_loss)
                    temperature_losses.append(temperature_loss)

                #updateQ
                if self.train_iter % self._config["frequency_update_Q"] == 0:
                    q_loss = self.calculate_critic_loss(s0,a,rew,done,s1)
                    q_loss = self.update_q_nets(q_loss)
                    q_losses.append(q_loss)


                #Update targets
                if self.train_iter % self._config["frequency_update_targets"] == 0:
                    self.target.soft_update(self.critic)

                self.train_iter +=1
        return q_losses,policy_losses,temperature_losses


def quantile_huber_loss(quantiles, samples,device="cpu"):
    # input_dim = batch,nets,quantiles,samples
    #https://en.wikipedia.org/wiki/Huber_loss
    #https://github.com/chainer/chainerrl/issues/590
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]
    abs_pairwise_delta = torch.abs(samples[:, None, None, :] - quantiles[:, :, :, None])
    kappa = 1
    huber_loss = torch.where(abs_pairwise_delta <= kappa,
                            pairwise_delta ** 2 * 0.5,
                            abs_pairwise_delta-0.5*kappa)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss/kappa).mean()
    return loss
