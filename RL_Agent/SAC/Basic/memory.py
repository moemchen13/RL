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
    

class SampleType(Enum):
    FINAL = 0
    FUTURE = 1
    EPISODE = 2
    RANDOM = 3

class HER_Memory(Memory):
    def __init__(self,state_dim,action_dim,reward=1,max_size=int(1e6),device='cpu',n_goals=4,goal_sampling=SampleType.FUTURE):
        super().__init__(state_dim,action_dim,max_size,device)
        self.start_episode = 0
        self.goal_sampling = goal_sampling
        #dont choose extra huge amount
        self.n_goals = n_goals
        self.reward = reward

    def create_hindsight_experience(self):
        if self.goal_sampling == SampleType.FUTURE:
            experiences = self.create_future_experience()
        if self.goal_sampling == SampleType.FINAL:
            experiences = self.create_final_experience()
        if self.goal_sampling == SampleType.EPISODE:
            experiences = self.create_episode_experience()
        if self.goal_sampling == SampleType.RANDOM:
                experiences = self.create_random_experience()
        for experience in experiences:
            self.add_transition(experience)
        self.start_episode = self.current_idx

    
    def create_future_experience(self):
        last_episode_entry = self.current_idx-self.n_goals
        # If we start again to fill replay buffer
        if self.start_episode > self.current_idx:
            last_episode_entry = self.size + self.current_idx -self.n_goals

        sample_index = np.random.randint(self.start_episode,last_episode_entry,1)
        new_experiences = []
        indexes = np.random.choice(np.arange(sample_index,last_episode_entry+self.n_goals),self.n_goals,replace=False)
        indexes = np.mod(indexes,self.max_size)

        for index in indexes:
            experience = [getattr(self,name)[index] for name in self.transition_names]
            new_experience = [entry.copy() for entry in experience]
            new_experience[self.transition_names.index('rew')] = self.reward
            new_experiences.append(new_experience)
            
        return new_experiences
    

    def create_final_experience(self):
        experience = [getattr(self,name)[self.current_idx-1] for name in self.transition_names]
        new_experience = [entry.copy() for entry in experience]
        new_experience[self.transition_names.index('rew')] = self.reward

        return list(new_experience)


    def create_episode_experience(self):
        last_episode_entry = self.current_idx
        # If we start again to fill replay buffer
        if self.start_episode > self.current_idx:
            last_episode_entry = self.size + self.current_idx

        new_experiences = []
        indexes = np.random.choice(np.arange(self.start_episode,last_episode_entry),self.n_goals,replace=False)
        indexes = np.mod(indexes,self.max_size)

        for index in indexes:
            experience = [getattr(self,name)[index] for name in self.transition_names]
            new_experience = [entry.copy() for entry in experience]
            new_experience[self.transition_names.index('rew')] = self.reward
            new_experiences.append(new_experience)
        return new_experiences


    def create_random_experience(self):
        
        new_experiences = []
        indexes = np.random.choice(np.arange(0,self.size),self.n_goals,replace=False)

        for index in indexes:
            experience = [getattr(self,name)[index] for name in self.transition_names]
            new_experience = [entry.copy() for entry in experience]
            new_experience[self.transition_names.index('rew')] = self.reward
            new_experiences.append(new_experience)
        return new_experiences

    def add_transition(self, transitions_new):
        for name,value in zip(self.transition_names,transitions_new):
            getattr(self,name)[self.current_idx] = value
            
        self.current_idx = (self.current_idx +1)%self.max_size #overwrite old entries
        self.size = min(self.size+1,self.max_size)
        
        terminal = bool(transitions_new[4])
        if terminal:
            self.create_hindsight_experience()




        
        
