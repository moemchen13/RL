import heapq

import numpy as np
import torch


# class to store transitions
class Memory():
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

    def get_all_transitions(self):
        return (torch.FloatTensor(getattr(self,name)[:]).to(self.device) for name in self.transition_names)

class PrioritizedMemory():
    def __init__(self,state_dim,action_dim,max_size=int(1e6),device='cpu'):
        self.device = device
        self.transition_names = ['s0','action','rew','s1','done']
        self.input_dims = [state_dim,action_dim,1,state_dim,1,1,]
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size
        self.heap = []

    def add_transition(self, transitions_new):
        #get new transition from environment
        s0,a,rew,s1,done = transitions_new
        if self.input_dims[0] != 1:
            s1 = s1[None,:]
            s0 = s0[None,:]
        if self.input_dims[1] != 1: 
            a = a[None,:]
        td = 0

        #overwrite old entries but check if earlier maximum
        self.current_idx = (self.current_idx +1)%self.max_size 
        self.size = min(self.size+1,self.max_size)

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        raise NotImplementedError("sample operation")
    
    def update_heap():
        raise NotImplementedError("Should update if we delete biggest elemnt update N in Heap or change alpha and beta")

    def calculate_weight():
        raise NotImplementedError("implement")


    def get_all_transitions(self):
        raise NotImplementedError("get all not necessary")
    