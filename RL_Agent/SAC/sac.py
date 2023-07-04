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
            "start_steps":10000,
            "eps": 0.1,
            "discount": 0.99,
            "buffer_size": int(1e7),
            "batch_size": 256,
            "lr_actor": float(3e-4),
            "lr_critic": float(1e-3),
            "lr_value": float(0.0001),
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
        self.memory = mem.Memory(max_size=self._config["buffer_size"])
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self.action_dim = action_space.shape[0]
        self.discount = self._config["discount"]
        self.tau = self._config["tau"]
        self.train_iter=0
        self.eval=False
        self.start_steps = self._config["start_steps"]
        if self._config["autotuned_temperature"]:
            self.target_entropy = -torch.Tensor(self.action_dim)
            self.log_temperature = torch.zeros(1,requires_grad=True)
            self.temperature = self.log_temperature.exp().item()
            self.temperature_optimizer = torch.optim.Adam([self.log_temperature],lr=self._config["lr_critic"])
        else:
            self.temperature = self._config["temperature"]


        self.actor = Actor(self._obs_dim,self.action_dim,action_space=action_space,hidden_sizes=self._config["hidden_size_actor"],
                            learning_rate=self._config["lr_actor"])
        self.critic_1 = NN.Feedforward(input_dim=self._obs_dim+self.action_dim,hidden_sizes=self._config["hidden_size_critic"]
                                    ,output_size=1,learning_rate=self._config["lr_critic"],
                                    name='critic_1')
        self.critic_2 = NN.Feedforward(input_dim=self._obs_dim+self.action_dim,hidden_sizes=self._config["hidden_size_critic"],
                                    output_size=1,learning_rate=self._config["lr_critic"],
                                    name='critic_2')
        self.target_critic_1 = NN.Feedforward(input_dim=self._obs_dim+self.action_dim,hidden_sizes=self._config["hidden_size_critic"]
                                    ,output_size=1,learning_rate=self._config["lr_critic"],
                                    name='target_critic_1')
        self.target_critic_2 = NN.Feedforward(input_dim=self._obs_dim+self.action_dim,hidden_sizes=self._config["hidden_size_critic"],
                                    output_size=1,learning_rate=self._config["lr_critic"],
                                    name='target_critic_2')
        self.update_network_targets(1)
        


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


    def eval(self):
        self.eval = True
    

    def train(self):
        self.eval = False


    def act(self,state):
        state = torch.from_numpy(state)
        if self.start_steps> self.memory.size:
            action = self.actor.random_action()
        else:
            if self.eval:
                action = self.actor.get_action(state)
            else:
                action, _ = self.actor.get_action_and_log_probs(state)
        return action.detach().numpy()

    
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
    

    def update_network_targets(self,tau=None):
        #update the first network
        if tau is None:
            tau = self.tau
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

    
    def update_Q_functions(self,s0,action,done,rew,s1):
        with torch.no_grad():
            a_next , log_prob_next = self.actor.get_action_and_log_probs(s1,reparameterize=False)
            min_Q_next = self.get_target_Q_value(s1,a_next)
            min_Q_next[done] = 0.0
            #get V estimate
            target_value = min_Q_next - self.temperature * log_prob_next
            
            y = (rew + self.discount * (1 - done)*target_value)
            #y = (rew + self.discount * (1 - done)*target_value).detach()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        Q_val_1 = self.critic_1.forward(torch.cat([s0,action],dim=1))
        Q_val_2 = self.critic_2.forward(torch.cat([s0,action],dim=1))
        
        #ToDO: find out if need of squeezing of qloss
        if self._config["use_smooth_L1"]:
            Q_loss_1 = 0.5* torch.nn.functional.smooth_l1_loss(Q_val_1,y,reduction='mean')
            Q_loss_2 = 0.5* torch.nn.functional.smooth_l1_loss(Q_val_2,y,reduction='mean')
        else:
            Q_loss_1 = 0.5* torch.nn.functional.mse_loss(Q_val_1,y)
            Q_loss_2 = 0.5* torch.nn.functional.mse_loss(Q_val_2,y)
        
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

        self.train_iter +=1
        
        for i in range(iter_fit):
            #print(i)
            if self.memory.size > self._config["batch_size"]:
                #Sample Batches from the replay Buffer
                data=self.memory.sample(batch=self._config["batch_size"])
                #Data Preparation: torchvectors
                s0 = to_torch(np.stack(data[:,0]))
                a = to_torch(np.stack(data[:,1]))
                rew = to_torch(np.stack(data[:,2])[:,None])
                s1 = to_torch(np.stack(data[:,3]))
                done = to_torch_int(np.stack(data[:,4])[:,None])
                
                ######Start SAC train loop#######
                #updateQ
                if i % self._config["frequency_update_Q"] == 0:
                    q_loss_1,q_loss_2 = self.update_Q_functions(s0,a,done,rew,s1)
                    q_losses.append(q_loss_1)
                    q_losses.append(q_loss_2)
                    f = open("q_loss.txt", "a")
                    f.write(str(q_loss_1))
                    f.write("       ")
                    f.write(str(q_loss_2))
                    f.write("\n")
                    f.close()


                #update policy
                if i % self._config["frequency_update_actor"] == 0:
                    action, log_prob = self.actor.get_action_and_log_probs(s0,reparameterize=True)
                    actor_Q = self.get_Q_value(s0,action)
                    actor_loss = -(actor_Q-self.temperature*log_prob).mean(axis=0)
                    #actor_loss = (self.temperature * log_prob - actor_Q).mean(axis=0)
                    actor_loss = self.update_policy(actor_loss)
                    policy_losses.append(actor_loss)
                    f = open("actor_loss.txt", "a")
                    f.write(str(actor_loss))
                    f.write("\n")
                    f.close()

                #Update temperture
                if self._config["autotuned_temperature"]:
                    temperature_loss = self.update_temperature(log_prob)
                else:
                    temperature_loss = torch.tensor(0.)

                #Update targets networks
                if i % self._config["frequency_update_targets"] == 0:
                    self.update_network_targets()

                
        return q_losses, policy_losses

