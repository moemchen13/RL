import heapq
from enum import Enum

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


class SampleType(Enum):
    FINAL = 0
    FUTURE = 1
    EPISODE = 2
    RANDOM = 3

class HER_Memory(Memory):
    def __init__(self,state_dim,action_dim,reward=1,max_size=int(1e6),device='cpu',extra_goals=1,goal_sampling=SampleType.FUTURE):
        super().__init__(state_dim,action_dim,max_size,device)
        self.start_episode = 0
        self.goal_sampling = goal_sampling
        self.extra_goals = extra_goals

    def create_hindsight_experience(self,episode_end):
        match self.goal_sampling:
            case SampleType.FUTURE:
                experiences = self.create_future_experience(episode_end)
            case SampleType.FINAL:
                experiences = self.create_final_experience(episode_end)
            case SampleType.EPISODE:
                experiences = self.create_episode_experience(episode_end)
            case SampleType.RANDOM:
                experiences = self.create_random_experience(episode_end)
        for experience in experiences:
            self.add_transition(experience)
        self.start_episode = self.current_idx+1 % self.max_size
    
    def create_future_experience(self,episode_end):
        NotImplementedError()

    def create_final_experience(self,epsiode_end):
        NotImplementedError()

    def create_episode_experience(self,episode_end):
        NotImplementedError()

    def create_random_experience(self,episode_end):
        NotImplementedError()



        
        
