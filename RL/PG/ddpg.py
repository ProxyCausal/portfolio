import torch
from torch import nn, optim
import gym
import os
from copy import deepcopy

from core import Net, Env
from utils import Logger, add_arguments

class Critic(Net):
    def __init__(self, sizes, activations):
        """
        Q(s, a=mu(s))
        """
        super().__init__(sizes, activations)
    
    def forward(self, s, a):
        """
        Args: 
            a: action from actor
        """
        q = torch.cat([s, a], dim=1)
        return super(q)

class DDPGNet(nn.Module):
    def __init__(self, sizes, activations):
        """Contains actor and critic

        Note: these terms are used differently than in VPG context
        """
        self.actor = Net(sizes, activations) #mu(s)
        self.critic = Critic(sizes, activations)
    
    def forward(self, s):
        a = self.actor(s)
        return self.critic(s, a)

def train(env_name, alr, clr, epochs, target, batch_size, seed=None):
    env = Env(env_name, seed)
    fname = os.path.splitext(os.path.basename(__file__))[0]
    logger = Logger(fname, seed)

    assert env.env.discrete, "Environment needs to have continuous action space"

    hidden_sizes = [256, 256]
    a_sizes = [env.state_dim] + hidden_sizes + [env.action_dim]
    v_sizes = [env.state_dim] + hidden_sizes + [1]
    activations = ([nn.Tanh()] * len(hidden_sizes)) + [nn.Identity()]
    ac = DDPGNet(a_sizes, activations) #has the uptodate params
    ac_target = deepcopy(ac)
    a_optim = optim.Adam(ac.actor.parameters(), lr=alr)
    c_optim = optim.Adam(ac.critic.parameters(), lr=clr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument('--alr', type=float, default=.01, help="Learning rate for actor")
    parser.add_argument('--clr', type=float, default=.01, help="Learning rate for critic")
    args = parser.parse_args()

    for i in range(args.repeat):
        train(args.env, args.alr, args.clr, args.epochs, args.target, args.batch_size, seed=args.seed)
