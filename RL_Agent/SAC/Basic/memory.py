import heapq
from enum import Enum

import numpy as np
import torch


# class to store transitions
class Memory(object):
    def __init__(self,state_dim,action_dim,max_size=int(1e6),device='cpu'):
        self.device = device
        self.transition_names = ('s0','action','rew','s1','done')
        self.input_dims = (state_dim,action_dim,1,state_dim,1)
        self.size =0
        self.current_idx = 0
        self.max_size=max_size
        for name, size in zip(self.transition_names,self.input_dims):
            setattr(self,name,np.empty((max_size,size)))

    def add_transition(self, transitions_new):
        for name,value in zip(self.transition_names,transitions_new):
            getattr(self,name)[self.current_idx] = value

        self.current_idx = (self.current_idx +1)%self.max_size #overwrite old entries
        self.size = min(self.size+1,self.max_size)

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        
        ind=np.random.choice(range(self.size), size=batch, replace=False)
        return (torch.FloatTensor(getattr(self,name)[ind]).to(self.device) for name in self.transition_names)
    

class PrioritizedReplay(object):
    
    def __init__(self, capacity, alpha=0.6,beta_start = 0.4,beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def beta_by_frame(self, frame_idx):
         
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done) 
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
    
    def sample(self, batch_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 

    def __len__(self):
        return len(self.buffer)
    
