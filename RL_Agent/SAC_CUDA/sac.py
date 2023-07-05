import os
import torch
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import pickle
from Basic import feedforward as NN
from Basic import memory as mem
from Agent import agent,UnsupportedSpace
from Actor import Actor
from Critic import Critic_Q

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class SAC_Agent(agent):
    def __init__(self,observation_space,action_space,device ="cpu", **userconfig):
        
        super().__init__(observation_space,action_space,device=device,**userconfig)
        
        self._config = {
            "start_steps":10,
            "eps": 0.1,
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
            "reward_scale":2,
            "update_target_every":1,
            "autotuned_temperature":False,
            "temperature":0.1,
            "use_smooth_L1":False,
            }
        self.device = device
        self.memory = mem.Memory(max_size=self._config["buffer_size"])
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self.action_dim = action_space.shape[0]
        self.discount = self._config["discount"]
        self.tau = self._config["tau"]
        self.train_iter=0
        self.eval_mode = False
        self.start_steps = self._config["start_steps"]
        if self._config["autotuned_temperature"]:
            self.target_entropy = -torch.Tensor(self.action_dim).to(self.device)
            self.log_temperature = torch.zeros(1,requires_grad=True,device=self.device)
            self.temperature = self.log_temperature.exp().item()
            self.temperature_optimizer = torch.optim.Adam([self.log_temperature],lr=self._config["lr_critic"])
        else:
            self.temperature = self._config["temperature"]


        self.actor = Actor(self._obs_dim,self.action_dim,action_space=action_space,hidden_sizes=self._config["hidden_size_actor"],
                            learning_rate=self._config["lr_actor"],device=self.device)
        self.critic = Critic_Q(self._obs_dim,self.action_dim,self._config["lr_critic"],hidden_sizes=self._config["hidden_size_critic"],device=self.device)

        self.target = Critic_Q(self._obs_dim,self.action_dim,self._config["lr_critic"],hidden_sizes=self._config["hidden_size_critic"],
                             tau=self.tau,target=True,device=self.device)
        
        self.target.soft_update(self.critic,tau=1)
        

    def store_transition(self,transition):
        self.memory.add_transition(transition)

    
    def get_networks_states(self):
        return (self.actor.state_dict(),
                self.critic.get_network_states(),
                self.target.get_network_states())
    

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


    def act(self,state):
        state = torch.from_numpy(state).to(self.actor.device)
        if self.eval_mode:
            action = self.actor.get_action(state)
        else:
            if self.start_steps> self.memory.size:
                action = self.actor.random_action()
            else:
                action, _ = self.actor.get_action_and_log_probs(state)
        return action.detach().cpu().numpy()

    
    def get_Q_value(self,state,action):
        Q_val = self.critic.get_min_Q_value(state,action)
        return Q_val
    

    def get_target_Q_value(self,state,action):
        #same as get_Q_value but from target networks
        target_Q_val = self.target.get_min_Q_value(state,action)
        return target_Q_val

    
    def update_Q_functions(self,s0,action,done,rew,s1):
        with torch.no_grad():
            a_next , log_prob_next = self.actor.get_action_and_log_probs(s1,reparameterize=False)
            min_Q_next = self.get_target_Q_value(s1,a_next)
            min_Q_next[done] = 0.0
            #get V estimate
            target_value = min_Q_next - self.temperature * log_prob_next
            
            y = (rew + self.discount * (1 - done)*target_value)
            #y = (rew + self.discount * (1 - done)*target_value).detach()

        q_loss = self.critic.update_critics(state=s0,action=action,target=y)
        return q_loss      


    def update_policy(self,s0):
        action, log_prob = self.actor.get_action_and_log_probs(s0,reparameterize=True)
        actor_Q = self.get_Q_value(s0,action)
        actor_loss = -(actor_Q-self.temperature*log_prob).mean(axis=0)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.item(), log_prob
    

    def update_temperature(self,log_probs):
        self.temperature_optimizer.zero_grad()
        temperature_loss  =  -(self.log_temperature * (log_probs+ self.target_entropy).detach()).mean()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.temperature = self.log_temperature.exp()
        return temperature_loss.item()


    def train(self,iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        to_torch_int = lambda x: torch.from_numpy(x.astype(np.int_))
        q_losses = []
        policy_losses = []
        temperature_losses=[]

        self.train_iter +=1
        
        for i in range(iter_fit):
            #print(i)
            if self.memory.size > self._config["batch_size"]:
                #Sample Batches from the replay Buffer
                data=self.memory.sample(batch=self._config["batch_size"])
                #Data Preparation: torchvectors
                s0 = to_torch(np.stack(data[:,0]),device=self.device)
                a = to_torch(np.stack(data[:,1]),device=self.device)
                rew = to_torch(np.stack(data[:,2])[:,None],device=self.device)
                s1 = to_torch(np.stack(data[:,3]),device=self.device)
                done = to_torch_int(np.stack(data[:,4])[:,None],device=self.device)
                
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

