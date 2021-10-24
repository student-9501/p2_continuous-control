"""
trains an agent on the reacher environment
"""
import os
import time
import argparse
from collections import deque

import numpy as np
import torch
import wandb

from agent import Agent
from mywand import Wand


def train(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=None,
          stop_score=None, env_id=None, project='reacher-bench',
          note='none', noise_stddev=None,
          filename='Reacher_Linux_NoVis/Reacher.x86_64', device=None,
          env=None):
    """
    :param note: a message to be added to the wandb data
    :param n_episodes: number of training episodes
    :param max_t: maximum number of timesteps per episode
    :param eps_start: starting value of epsilon, for scaling noise for exploration
    :param eps_end: minimum value of epsilon
    :param eps_decay: decay rate of epsilon per episode
    :param noise_stddev: the amount of gaussian noise to be added to
        the actions for exploration
    :param stop_score: if set, stop as soon as the agent hits a threshold
        score averaged over 100 episodes
    :param project: the project name to log runs to in weights and biases
    :param env_id: if set, launch unity on a new port to run parallel sessions
    :param filename: the unity binary to run, eg. 'Banana_Linux_NoVis/Banana.x86_64'
    :param device: the device to run the model on, eg. 'cpu', 'cuda:0', 'cuda:1', etc
    :param env: a unity environment to use, if none is specified a new
        one will be created

    :returns: the unity environment, so it can be reused properly if we
        are running a benchmark
    """
    from reacher_env import Environment
    if env is None:
        print('creating new env')
        env = Environment(env_id=env_id, filename=filename)
    else:
        print('reusing old env', env)
    agent = Agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            device_name=device,
            noise_stddev=noise_stddev,
    )
    wand = Wand(agent)
    wand.login()

    if os.environ.get('WANDB_ENTITY'):
        wand.init(project=project, entity=os.environ.get('WANDB_ENTITY'), group=note, config={
            'LR_ACTOR': agent.lr_actor,
            'LR_CRITIC': agent.lr_critic,
            'TAU': agent.tau,
            'HIDDEN_SIZE_1': agent.hidden_size_1,
            'HIDDEN_SIZE_2': agent.hidden_size_2,
            'BUFFER_SIZE': agent.buffer_size,
            'NOISE_STDDEV': agent.noise_stddev,
            'GAMMA': agent.gamma,
            'NOTE': note,
            'EPS_DECAY': eps_decay,
        })

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(np.array([action]))
            agent.step(state, action, reward, next_state, done)
            agent.learn_explicit()
            state = next_state
            score += reward
            if done:
                print('weird, done?', done, reward)
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        wand.log({
            "scores": np.mean(scores_window),
            "eps": eps,
        })
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} eps: {:.6f}'.format(
            i_episode, np.mean(scores_window), eps), end="")
        if i_episode % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if stop_score is not None and np.mean(scores_window) >= stop_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    else:
        torch.save(agent.local_actor.state_dict(), 'actor-final.pth')
        torch.save(agent.local_critic.state_dict(), 'critic-final.pth')
    print('done training')
    wandb.finish()
    return env


def run_args(args, override):
    train_args = dict(
        n_episodes=args.n_episodes,
        max_t=args.max_t,
        env_id=args.env_id,
        eps_decay=args.eps_decay,
        filename=args.filename,
        device=args.device,
        note=args.note,
        noise_stddev=args.noise_stddev,
        stop_score=args.stop_score,
    )
    for k, v in override.items():
        train_args[k] = v
    print('running:', train_args)
    return train(**train_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_score', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--env_id', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=250)
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--note', type=str, default='none')
    parser.add_argument('--eps_decay', type=float, default=1)
    parser.add_argument('--noise_stddev', type=float, default=0.85)
    parser.add_argument('--filename', type=str, default='Reacher_Linux_NoVis/Reacher.x86_64')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark_iter', type=int, default=1)
    args = parser.parse_args()
    if args.benchmark:
        # if set, run these settings run <benchmark_iter> times
        # with 5 parallel training jobs per GPU, in order to get
        # a more stable measurement of the average training result
        env_id = 0
        overrides = []
        for device in ['cuda:0', 'cuda:1']:
            for i in range(5):
                env_id += 1
                overrides.append({
                    'env_id': env_id,
                    'device': device,
                })
        pids = []
        for override in overrides:
            pid = os.fork()
            if pid == 0:
                os.setsid()
                env = None
                override['env'] = env
                for i in range(args.benchmark_iter):
                    env = run_args(args, override)
                    override['env'] = env
                os._exit(0)
            else:
                pids.append(pid)
                time.sleep(2)
        print('waiting for children')
        for pid in pids:
            os.waitpid(pid, 0)
            print('waited for', pid)
    else:
        run_args(args, {})
        os._exit(0)  # workaround for unity cleanup code hanging sometimes


if __name__ == '__main__':
    main()
