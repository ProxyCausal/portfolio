import torch
from torch import nn
import gym
from gym.spaces import Discrete, Box
import numpy as np

class Net(nn.Module):
    def __init__(self, sizes, activations, output=False):
        """Basic feedforward network

        Args:
            sizes: a list of input, hidden, and output sizes,
            should have at least length 2
            activations: a list of torch activation functions to apply after each layer
            output: bool indicating to skip last layer
        """
        super().__init__()
        layers = []
        for i in range(len(sizes)-1):
            if i == len(sizes)-2 and output:
                break
            layers += [nn.Linear(sizes[i], sizes[i+1])]

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activations[i](self.layers[i](x))
        return x

class Env(object):
    def __init__(self, env_name, seed=None):
        self.env = gym.make(env_name)
        #observations are assumed to be continuous
        if isinstance(self.env.action_space, Discrete):
            self.discrete = True
            #number of actions to choose from, not dim of action vector
            self.n_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            self.discrete = False
            self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

        #seed for reproducability
        self.seed = np.random.randint(10000) if seed is None else seed 
        self.env.seed(self.seed)
        torch.manual_seed(self.seed)