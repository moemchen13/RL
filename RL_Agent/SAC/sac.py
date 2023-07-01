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
            "eps": 0.1,
            "discount": 0.99,
            "buffer_size": int(1e7),
            "batch_size": 256,
            "lr_actor": float(0.00001),
            "lr_critic": float(0.0001),
            "lr_value": float(0.0001),
            "hidden_size_critic": [256,256],
            "hidden_size_actor": [256,256],
            "hidden_size_value": [256,256],
            "tau": 0.005,
            "reward_scale":2,
            "update_target_every":1,
            "temperature":0.1,
        }
        self.memory = mem.Memory(max_size=self._config["buffer_size"])
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._n_actions = action_space.shape[0]
        self.discount = self._config["discount"]
        self.tau = self._config["tau"]
        self.train_iter=0

        high,low = torch.from_numpy(self._action_space.high), torch.from_numpy(self._action_space.low)

        self.actor = Actor(self._obs_dim,self._n_actions,max_action=high,hidden_sizes=self._config["hidden_size_actor"],
                            learning_rate=self._config["lr_actor"])
        self.critic_1 = NN.Feedforward(input_dim=self._obs_dim+self._n_actions,hidden_sizes=self._config["hidden_size_critic"]
                                    ,output_size=self._n_actions,learning_rate=self._config["lr_critic"],
                                    name='critic_1')
        self.critic_2 = NN.Feedforward(input_dim=self._obs_dim+self._n_actions,hidden_sizes=self._config["hidden_size_critic"],
                                    output_size=self._n_actions,learning_rate=self._config["lr_critic"],
                                    name='critic_2')
        self.target_critic_1 = NN.Feedforward(input_dim=self._obs_dim+self._n_actions,hidden_sizes=self._config["hidden_size_critic"]
                                    ,output_size=self._n_actions,learning_rate=self._config["lr_critic"],
                                    name='target_critic_1')
        self.target_critic_2 = NN.Feedforward(input_dim=self._obs_dim+self._n_actions,hidden_sizes=self._config["hidden_size_critic"],
                                    output_size=self._n_actions,learning_rate=self._config["lr_critic"],
                                    name='target_critic_2')
        self.update_network_parameters()
        

    def act(self,state):
        state = torch.from_numpy(state)
        action = self.actor.get_action(state)
        return action.detach().numpy()

    def update_network_parameters(self,tau=None):
        if tau is None:
            tau = self.tau
        #update the first network
        target_value_params = self.target_critic_1.named_parameters()
        value_params = self.critic_1.named_parameters()
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() 
            + (1-tau)*target_value_state_dict[name].clone()
        self.target_critic_1.load_state_dict(value_state_dict)

        #Update the second network
        target_value_params = self.target_critic_2.named_parameters()
        value_params = self.critic_2.named_parameters()
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() 
            + (1-tau)*target_value_state_dict[name].clone()
        self.target_critic_2.load_state_dict(value_state_dict)

    def store_transition(self,transition):
        self.memory.add_transition(transition)
    
    def get_networks_states(self):
        return (self.actor.state_dict(), self.critic_1.state_dict(),
                self.critic_2.state_dict(),self.target_critic_1.state_dict()
                ,self.target_critic_2.state_dict())

    def load_network_states(self,state):
        self.actor.load_state_dict(state[0])
        self.critic_1.load_state_dict(state[1])
        self.critic_2.load_state_dict(state[2])
        self.target_critic_1.load_state_dict(state[3])
        self.target_critic_2.load_state_dict(state[4]) 
    
    def reset(self):
        pass
    
    def update_Q_functions(self,state,action,y):
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        Q_val_1 = self.critic_1.forward(torch.cat([state,action],dim=1))
        Q_val_2 = self.critic_2.forward(torch.cat([state,action],dim=1))
        mse_loss_1 = torch.nn.MSELoss()
        mse_loss_2 = torch.nn.MSELoss()
        Q_loss_1 = 0.5* mse_loss_1(Q_val_1,y)
        Q_loss_2 = 0.5* mse_loss_2(Q_val_2,y)
        Q_loss_1.backward()
        Q_loss_2.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        return Q_loss_1.item(),Q_loss_2.item()        

    def update_policy(self,actor_loss):
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.item()
    
    def get_Q_value(self,state,action):
        q1_value = self.critic_1.forward(torch.cat([state,action],dim=1))
        q2_value = self.critic_2.forward(torch.cat([state,action],dim=1))
        Q_val = torch.min(q1_value,q2_value)
        return Q_val
    
    def get_target_Q_value(self,state,action):
        #same as get_Q_value but from target networks
        target_q1_value = self.target_critic_1.forward(torch.cat([state,action],dim=1))
        target_q2_value = self.target_critic_2.forward(torch.cat([state,action],dim=1))
        target_Q_val = torch.min(target_q1_value,target_q2_value)
        return target_Q_val

    def train(self,iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        to_torch_int = lambda x: torch.from_numpy(x.astype(np.int_))
        losses = []

        self.train_iter +=1
        
        temperature = self._config["temperature"]
        
        for i in range(iter_fit):
            #print(i)
            if self.memory.size < self._config["batch_size"]:
                #Sample Batches from the replay Buffer
                data=self.memory.sample(batch=self._config["batch_size"])
                #Data Preparation: torchvectors
                s0 = to_torch(np.stack(data[:,0]))
                a = to_torch(np.stack(data[:,1]))
                rew = to_torch(np.stack(data[:,2])[:,None])
                s1 = to_torch(np.stack(data[:,3]))
                done = to_torch_int(np.stack(data[:,4])[:,None])
                
                ######Start SAC train loop#######
                a_next , log_prob_a_next = self.actor.get_action_and_log_probs(s1)
                min_Q_next = self.get_target_Q_value(s1,a_next)
                min_Q_next[done] = 0.0

                y = (rew + self.discount * (1 - done)*(min_Q_next - temperature* log_prob_a_next)).detach()
                q_loss_1,q_loss_2 = self.update_Q_functions(s0,a,y)
                ##### Updated Q Functions ########

                #torch.autograd.set_detect_anomaly(True)
                '''
                a_reparametrized, log_probs_reparameterized = self.actor.get_action_and_log_probs(s0,reparameterize=True)
                min_Q = self.get_Q_value(s0,a_reparametrized).detach()
                actor_loss = (temperature* log_probs_reparameterized - min_Q).mean()
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                policy_loss = actor_loss.item()
                '''
                #policy_loss = self.update_policy(actor_loss)
                dist = self.actor.get_dist(s0)
                action = dist.rsample()
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                actor_Q = self.get_Q_value(s0,action)
                actor_loss = (temperature * log_prob - actor_Q).mean()

                # optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                ##### Updated Policy ##########

                self.update_network_parameters()

                losses.append((q_loss_1,q_loss_2,actor_loss.item()))
        return losses

