import torch
import numpy as np
from gymnasium import spaces
from Basic import memory as mem
from abc import ABC, abstractmethod


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class agent(object):
    def __init__(self,observation_space,action_space, **userconfig):
        super(agent,self).__init__()
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))
        self._config= {}
        self.memory = None
        self.discount = None
        self.action_space = action_space
        self._n_actions = action_space.shape[0]
    
    @abstractmethod
    def act(self,state):
        pass

    @abstractmethod
    def store_transition(self,transition):
        pass
    
    @abstractmethod
    def get_networks_states(self):
        pass

    @abstractmethod
    def load_network_states(self, states):
        pass

    @abstractmethod
    def act(self,state):
        pass

    @abstractmethod
    def train(self):
        pass

