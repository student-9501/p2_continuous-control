"""
implements a DDPG agent to play the reacher game.
"""
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim

from model import ActorModel, CriticModel
from history import History, Sample


# default parameters that can be overridden in the agent's constructor
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
STATE_SIZE = 33
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 256
GAMMA = 0.95
LEARN_EVERY = 20
LEARN_CYCLES = 10
ACTION_SIZE = 4


class Agent:
    local_actor: Callable = None

    def __init__(self, state_size: int = 33, action_size: int = 4,
                 seed: int = 0, batch_size: int = BATCH_SIZE,
                 buffer_size: int = BUFFER_SIZE, hidden_size_1: int = HIDDEN_SIZE_1,
                 hidden_size_2: int = HIDDEN_SIZE_2,
                 lr_actor: float = LR_ACTOR, lr_critic: float = LR_CRITIC,
                 gamma: float = GAMMA, tau: float = TAU,
                 learn_every=LEARN_EVERY, device_name: str = None,
                 noise_stddev: float = 1):
        """
        :param state_size: the size of the state space as the number of floats
        :param action_size: the size of the action space as a number
            of floating point values
        :param seed: used to initialize a new model
        :param noise_stddev: the standard deviation of the gaussian
            noise added to the action vector during training
        :param learn_every: the number of steps between learning
        :param batch_size: how many episodes to sample per learning step
        :param buffer_size: how many episodes to keep in the replay
            buffer before discarding episodes
        :param hidden_size_1: this size of the first layer in the neural network
        :param hidden_size_2: this size of the second layer in the neural network
            of the neural network
        :param lr_actor: the learning rate for the actor's optimizer
        :param lr_critic: the learning rate for the critic's optimizer
        :param gamma: the discount factor for transfering future reward
            back down the chain
        :param tau: the rate at which we transfer the latest model
            weights to the target network
        :param device_name: cpu, cuda:0, etc.  if null, default to
            either cpu or cuda:0, used for running parallel simulations
            on a multi-GPU instance
        """
        # super().__init__(state_size, action_size, seed)
        if not seed:
            seed = np.random.randint(0, 100000)
        if device_name is None:
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.batch_size = batch_size
        self.noise_stddev = noise_stddev
        self.learn_c = 0
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.learn_every = learn_every
        model_args = {
            'state_size': state_size,
            'action_size': action_size,
            'seed': seed,
            'hidden_size_1': hidden_size_1,
            'hidden_size_2': hidden_size_2,
        }
        self.local_actor = ActorModel(**model_args).to(self.device)
        self.target_actor = ActorModel(**model_args).to(self.device)
        self.local_critic = CriticModel(**model_args).to(self.device)
        self.target_critic = CriticModel(**model_args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.local_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.local_critic.parameters(), lr=lr_critic)
        self.history = History(buffer_size)

        self.batches_learned = 0

    def act(self, state, eps):
        """
        :param state: a 33 wide tensor representing the current game state
        :param eps: float from [0, 1] which acts as a multiplier to scale
            down the amount of noise as training progresses
        :returns: an array of 4 floats
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_actor.eval()
        with torch.no_grad():
            actions = self.local_actor(state).squeeze(0).detach().cpu().numpy()
            actions += np.random.normal(0, self.noise_stddev, self.action_size) * eps
            actions = np.clip(actions, -1, 1)
        self.local_actor.train()

        return actions

    def step(self, state: list[float], action: int, reward: float, next_state: list[float], done: bool):
        """ store a sample in the buffer, and do nothing else.  for
        distributed experience gathering, that's all that's necessary
        and learning is handled separately in the the learn method.
        :param state: the state which we determined this action from
        :param action: the action we took in this experience
        :param reward: the reward received from this exact point
        :param next_state: an array of floats representing the next state
        :param done: whether this is a terminal state for an episode,
            which signals that we can discard future Q values when
            calculating the expected value
        """
        sample = Sample(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.history.add_sample(sample)

    def learn_explicit(self):
        """ typically called after step(), every `learn_every` steps and
        once we have sufficient samples in our history, try to improve
        our behaviour model by taking a random sample and applying the
        iterative update algorithm
        """
        if self.learn_c % self.learn_every == 0 and len(self.history) >= BATCH_SIZE:
            for _ in range(LEARN_CYCLES):
                batch = self.history.random_sample(self.batch_size)
                self.learn_batch(batch)
        self.learn_c += 1

    def learn_batch(self, batch: list[Sample]):
        """ take a batch of samples, calculate the TD error, and move
        the local network closer to the optimal function

        :param batch: a list of samples which we pulled from the replay buffer
        """
        self.batches_learned += 1
        rewards = torch.Tensor([[x.reward] for x in batch]).to(self.device)
        done = torch.Tensor([[x.done] for x in batch]).to(self.device)
        actions = torch.Tensor([x.action for x in batch]).to(self.device)
        next_states = torch.Tensor([x.next_state for x in batch]).to(self.device)
        states = torch.Tensor([x.state for x in batch]).to(self.device)

        # update critic
        target_action = self.target_actor(next_states)
        target_q_next = self.target_critic(next_states, target_action)
        target_q = rewards + (self.gamma * target_q_next * (1 - done))
        local_q = self.local_critic(states, actions)
        criterion = nn.MSELoss()
        loss = criterion(local_q, target_q)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        # update actor
        pred_actions = self.local_actor(states)
        local_q = -self.local_critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        local_q.backward()
        self.actor_optimizer.step()
        self.soft_update(self.local_actor, self.target_actor)
        self.soft_update(self.local_critic, self.target_critic)

    def soft_update(self, local, target):
        """
        move the target network slightly closer to the local network.
        the original algorithm synced the target model with the local
        model all at once periodically, while soft updates have a
        smoother transition.  the rate at which we adjust the target
        network is controlled by the `tau` parameter.

        :param local: either the local actor or critic, weights are partially copied from
        :param target: either the target actor or critic, weights are partially copied to this
        """
        with torch.no_grad():
            for source, dest in zip(local.parameters(), target.parameters()):
                # could use source.data and dest.data instead of source/dest?
                dest.data.copy_(dest.data * (1 - self.tau) + source.data * self.tau)
