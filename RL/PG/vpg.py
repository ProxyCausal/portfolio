import torch
from torch import nn, optim
#make sure this is not multivariate!! which would make the mem usage spike, why?
from torch.distributions import Normal 
from torch.distributions.categorical import Categorical
import gym
import numpy as np
import os

from core import Net, Env
from utils import Logger, add_arguments

class VPGNet(Net):
    def __init__(self, sizes, activations, m, discrete):
        """Subclass of Net

        Args:
            m: a string indicating policy or value
            discrete: bool indicating action space is continuous or discrete

        Returns:
            If discrete, softmax probabilities of size (batch, output_size)
            If continuous, mean and variances for normal
        """
        #if continuous specify output layer
        super().__init__(sizes, activations, output=not discrete)
        self.m = m
        self.discrete = discrete
        if not self.discrete:
            self.out_mu = nn.Linear(sizes[-2], sizes[-1])
            self.out_sigma = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        x = super().forward(x)
        if self.m == 'policy':
            x = torch.softmax(x, dim=1)
        if not self.discrete:
            x = mu, log_sigma_sq = self.out_mu(x), self.out_sigma(x)
        return x

def train(env_name, mode, alr, vlr, epochs, target, batch_size, seed=None):
    """Trains policy using gradients via offline learning

    Args:
        batch_size is # of time steps
    """
    env = Env(env_name, seed)
    fname = os.path.splitext(os.path.basename(__file__))[0]
    logger = Logger(fname, seed)

    hidden_sizes = [32]
    a_sizes = [env.state_dim] + hidden_sizes + [env.n_actions if env.discrete else env.action_dim]
    v_sizes = [env.state_dim] + hidden_sizes + [1]
    activations = ([nn.Tanh()] * len(hidden_sizes)) + [nn.Identity()]
    policy = VPGNet(a_sizes, activations, 'policy', env.discrete)
    baseline = VPGNet(v_sizes, activations, 'value', env.discrete)
    a_optim = optim.Adam(policy.parameters(), lr=alr)
    v_optim = optim.Adam(baseline.parameters(), lr=vlr)

    evaluator = True if mode == 'baseline' or mode == 'critic' else False

    def get_policy(obs):
        """Creates a (batch of) policy distributions

        Distribution is diagonal/isotropic gaussian for continuous,
        categorical for discrete

        Args:
            obs: tensor of size (batch, obs_dim)
        """
        if env.discrete:
            probs = policy(obs) #(batch, n_actions)
            policy_dist = Categorical(probs)
        else:
            mu, log_sigma_sq = policy(obs)
            sigma_sq = torch.exp(log_sigma_sq / 2) #(batch, dim_act)
            policy_dist = Normal(mu, sigma_sq)
        return policy_dist
    
    def cum_reward(rewards):
        """Cumulative reward in each episode after action at time step t
        """
        n = len(rewards)
        returns = [0] * n
        for i in reversed(range(n)):
            returns[i] = rewards[i] + returns[i+1] if i+1 < n else 0
        return returns

    def actor_loss(obs, actions, weights, values_s, values_s1):
        """Calculates loss derived from PG theorem

        Call the loss phi:
        it can also be the cum reward after action, optionally minus the baseline,
        the advantage function

        Args:
            Accepts args as tensors
            weights: If vanilla or baseline uses cumulative returns 
            If critic uses one-step returns
            values_s: Current state value approximated by value net
            values_s1: Next state value (shifted over by 1) approximated by value net
        """
        policy_dist = get_policy(obs)
        #for each cat dist get log pdf evaluated at chosen action of size (action_dim,)
        logp = policy_dist.log_prob(actions) #(batch, action_dim)
        delta = None
        if values_s is None:
            loss = -(logp * weights).mean() #minimize neg value of start state
        else:
            if mode == 'baseline':
                loss = -(logp * (weights - values_s)).mean()
            elif mode == 'critic':
                delta = ((weights+values_s1) - values_s)
                loss = -(logp * delta).mean() #use TD-0 instead of actual returns
        return loss, delta

    def value_loss(values_s, delta, weights):
        """Calculates loss for state value net
        """
        if delta is None:
            loss = ((values_s - weights)**2).mean() #minimize MSE
        else:
            loss = (delta ** 2).mean()
        return loss

    #inner function uses env, policy vars, etc...
    def train_one_epoch():
        """
        Loss function used is the analytically derived expression g_hat in 
        https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#id2
        without the gradient, except it only counts reward after action (MC or "R2G")
        """
        batch_actions = []
        batch_obs = []
        batch_1returns = [] #one step return after action at each time step t
        batch_returns = [] #cumulative reward in each ep after action at time step t
        batch_totals = [] #total return for each episode
        batch_lens = []

        eps_returns = []
        state = env.env.reset()
        while True:
            batch_obs.append(state)
            #get current parameterized policy distribution
            policy = get_policy(torch.Tensor(state).unsqueeze(0))
            #sample a single action
            action = policy.sample().item()
            batch_actions.append(action)
            state, reward, done, _ = env.env.step(action if env.discrete else [action])

            eps_returns.append(reward)
            
            if done:
                eps_return = sum(eps_returns)
                eps_len = len(eps_returns)
                batch_totals.append(eps_return)
                batch_lens.append(eps_len)
                batch_1returns.extend(eps_returns)
                batch_returns.extend(cum_reward(eps_returns))

                #reset eps vars
                eps_returns = []
                #continue collecting until batch is done
                state = env.env.reset()
                if sum(batch_lens) > batch_size:
                    break
             
        values_s = None
        if evaluator:
            values_s = baseline(torch.Tensor(batch_obs))
        values_s1 = None
        if mode == 'critic':
            values_s1 = baseline(torch.Tensor(batch_obs[1:]))
            #need to set end of eps to 0
            values_s1[torch.LongTensor(batch_lens[:-1])-1] = 0
            values_s1 = torch.cat((values_s1, torch.zeros(1).unsqueeze(1)))

        a_loss, delta = actor_loss(
            obs=torch.Tensor(batch_obs), #converts list of np arrays
            actions=torch.Tensor(batch_actions), 
            weights=torch.Tensor(batch_1returns) if mode == 'critic' else torch.Tensor(batch_returns),
            values_s=values_s,
            values_s1=values_s1,
            )
        a_optim.zero_grad()
        if evaluator:
            #graph should stay the same, also delta doesn't rely on a's params
            a_loss.backward(retain_graph=True)
        else: #need to free the graph
            a_loss.backward()
        a_optim.step()

        if evaluator:
            v_loss = value_loss(
                values_s=values_s,
                delta=delta,
                weights=torch.Tensor(batch_returns)
            )
            v_optim.zero_grad()
            v_loss.backward()
            v_optim.step()

        return a_loss, batch_totals, batch_lens

    for e in range(epochs):
        a_loss, batch_totals, batch_lens = train_one_epoch()
        print("epoch: %d \t a_loss: %.3f \t return: %.3f \t ep_len: %.3f" %
        (e, a_loss, np.mean(batch_totals), np.mean(batch_lens)))
        if np.mean(batch_totals) >= target:
            break
    
    logger.save_score(np.mean(batch_totals), e)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument('--alr', type=float, default=.01, help="Learning rate for actor")
    parser.add_argument('--vlr', type=float, default=.01, help="Learning rate for baseline")
    parser.add_argument('--mode', type=str, default='vanilla', help="Choose from vanilla, baseline, critic")
    args = parser.parse_args()

    for i in range(args.repeat):
        train(args.env, args.mode, args.alr, args.vlr, args.epochs, 
        args.target, args.batch_size, seed=args.seed)

"""
python vpg.py --target 300 --env CartPole-v1
python vpg.py --target 500 --env MountainCarContinuous-v0
"""