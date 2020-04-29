import gym
import argparse
import os
import torch
import numpy as np
import config
import env
from model import Controller

MODEL_DIR = 'models'

class RolloutGenerator(object):
    def __init__(self, env_name, render=False):
        super().__init__()
        self.env_prop = config.envs[env_name]
        self.render = render
        #self.env = gym.make(self.env_prop['NAME']) #doesn't render this way for some reason
        self.env = env.make_env(env_name, render)
        self.controller = Controller(
            self.env_prop['INPUT_DIM'], self.env_prop['HIDDEN_DIMS'], self.env_prop['ACTION_DIM'], self.env_prop['ACTIVATION'])

    def rollout(self, params=None):
        """
        Args: 
            params: parameters as a single (D,) np array
        """
        if params is not None:
            self.controller.set_parameters(params)
            assert np.allclose(params, self.controller.flat_parameters()) #check params were indeed set

        if self.render:
            self.env.render()

        #device = torch.device('cuda')
        #self.controller = self.controller.to(device)

        state = self.env.reset()
        eps_reward = 0
        for _ in range(self.env_prop['MAX_FRAMES']):
            action = self.controller(torch.Tensor(state).view(1,-1)).squeeze().detach().cpu().numpy()
            state, reward, done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            eps_reward += reward
            if done:
                break

        return -eps_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help="Filename of saved tar model")
    args = parser.parse_args()

    env_name = args.path.split('_')[0]

    generator = RolloutGenerator(env_name, render=True)
    params = torch.load(os.path.join(MODEL_DIR, args.path))
    generator.controller.load_state_dict(params['model_state_dict'])
    print(generator.rollout())
    #python rollout.py --env racecar