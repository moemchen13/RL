import os
import torch
import numpy as np
from feedforward import Feedforward
from Actor import Actor
from memory import Memory
from gymnasium import spaces
import gymnasium as gym
import pickle
import memory as mem

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class SAC_Agent(object):
    def __init__(self,observation_space,action_space, **userconfig):
        
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

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
            "use_target":True,
            "update_target_every":1,
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
        self.critic_1 = Feedforward(input_dim=self._obs_dim+self._n_actions,hidden_sizes=self._config["hidden_size_critic"]
                                    ,output_size=self._n_actions,learning_rate=self._config["lr_critic"],
                                    name='critic_1')
        self.critic_2 = Feedforward(input_dim=self._obs_dim+self._n_actions,hidden_sizes=self._config["hidden_size_critic"],
                                    output_size=self._n_actions,learning_rate=self._config["lr_critic"],
                                    name='critic_2')
        self.value = Feedforward(input_dim=self._obs_dim,hidden_sizes=self._config["hidden_size_value"],
                                 output_size=1,learning_rate=self._config["lr_value"],name='value')
        self.target_value = Feedforward(input_dim=self._obs_dim,hidden_sizes=self._config["hidden_size_value"],
                                        output_size=1,learning_rate=self._config["lr_value"],name='target_val')
        
        self.scale = self._config["reward_scale"]
        self.update_network_parameters()
        

    def act(self,state):
        state = torch.from_numpy(state)
        action,_ = self.actor.sample_normal(state,reparameterize=False)
        return action.detach().numpy()

    def update_network_parameters(self,tau=None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() 
            + (1-tau)*target_value_state_dict[name].clone()
        
        self.target_value.load_state_dict(value_state_dict)

    def copy_nets(self):
        self.target_value.load_state_dict(self.value.state_dict())
    
    def store_transition(self,transition):
        self.memory.add_transition(transition)
    
    def state(self):
        return (self.actor.state_dict(), self.critic_1.state_dict(),
                self.critic_2.state_dict(),self.value.state_dict(),self.target_value.state_dict())

    def restore_state(self,state):
        self.actor.load_state_dict(state[0])
        self.critic_1.load_state_dict(state[1])
        self.critic_2.load_state_dict(state[2])
        self.value.load_state_dict(state[3])
        self.target_value.load_state_dict(state[4]) 

    def evaluate_critic(self,state,reparameterized: bool):
        actions, log_probs = self.actor.sample_normal(state,reparameterize=reparameterized)
        q1_policy = self.critic_1.forward(torch.cat([state,actions],dim=1))
        q2_policy = self.critic_2.forward(torch.cat([state,actions],dim=1))
        ##Overestimation Bias correction
        critic_value = torch.min(q1_policy,q2_policy)
        return critic_value, log_probs
    
    def reset(self):
        pass

    def train(self,iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        to_torch_bool = lambda x: torch.from_numpy(x.astype(np.bool_))
        losses = []

        self.train_iter +=1
        if self._config["use_target"] and self.train_iter % self._config["update_target_every"] == 0:
            self.copy_nets()
        
        for i in range(iter_fit):
            #print(i)
            if self.memory.size < self._config["batch_size"]:
                    
                data=self.memory.sample(batch=self._config["batch_size"])
                s_t0 = to_torch(np.stack(data[:,0]))
                a = to_torch(np.stack(data[:,1]))
                rew = to_torch(np.stack(data[:,2])[:,None])
                s_t1 = to_torch(np.stack(data[:,3]))
                done = to_torch_bool(np.stack(data[:,4])[:,None])
                

                value = self.value(s_t0)
                new_value = self.target_value(s_t1)
                new_value[done] = 0.0

                ######Start SAC#######
                #update value
                #print("update value")
                #print(s_t0)
                critic_value, log_probs = self.evaluate_critic(s_t0,False)
                self.value.optimizer.zero_grad()
                value_target = critic_value - log_probs
                value_loss = 0.5 * self.value.loss(value,value_target)
                value_loss.backward()
                self.value.optimizer.step()

                #update actor     
                #print("update actor") 
                critic_value,log_probs= self.evaluate_critic(s_t0,True)
                actor_loss = log_probs - critic_value
                actor_loss = torch.mean(actor_loss)
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                #update critics
                #print("update critic")
                self.critic_1.optimizer.zero_grad()
                self.critic_2.optimizer.zero_grad()
                q_hat = self.scale*rew+self.discount*new_value #Bellmann Equation
                q1_old_policy = self.critic_1.forward(torch.cat([s_t0,a],dim=1))
                q2_old_policy = self.critic_2.forward(torch.cat([s_t0,a],dim=1))
                critic1_loss =0.5*self.critic_1.loss(q1_old_policy,q_hat)
                critic2_loss =0.5*self.critic_2.loss(q2_old_policy,q_hat)

                critic_loss = critic1_loss + critic2_loss
                critic_loss.backward()
                self.critic_1.optimizer.step()
                self.critic_2.optimizer.step()

                self.update_network_parameters()

                losses.append((critic_loss.item(),actor_loss.item()))
        return losses

