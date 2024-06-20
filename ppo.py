import argparse
import os
from distutils.util import strtobool
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import time
from lightning import seed_everything
import torch

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import numpy as np


#delete warnings
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parse = argparse.ArgumentParser()
    # environment setting
    parse.add_argument('--seed', type=int, default=2024, help='random seed for reproducibility')
    parse.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, help='use cuda or not')
    parse.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), 
                       default=True, help='whether to set `torch.backends.cudnn.deterministic=True`')
    parse.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, help="whether to log the model", const=True, nargs='?')
    parse.add_argument("--wandb-project-name", type=str, default="CleanRL", help="the project name of wandb")
    parse.add_argument("--wandb-entity", type=str, default=None, help="the entity of wandb")
    
    # experiment setting
    parse.add_argument('--exp-name', type=str, default=os.path.basename(__file__).split('.py')[0]
                       ,help='the name of this experiment')
    parse.add_argument("--gym-id", default="CartPole-v0", help="the id of the gym environment")
    parse.add_argument("--learning-rate", type=float, default=0.01, help="the learning rate of the optimizer")
    parse.add_argument("--total-timesteps", type=int, default=50000, help="total timesteps of the experiments")
    parse.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, help="whether to capture video")
    parse.add_argument("--n-envs", type=int, default=4, help="the number of parallel environments")
    parse.add_argument("--n-steps", type=int, default=128, help="the number of steps to run for each environment per update, aka the number of data to collect")

    return parse.parse_args()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # form the original paper https://arxiv.org/pdf/1707.06347
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critc = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01), # the std=0.01 make the the probability of each action is similar
        )

if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_name = f'{args.exp_name}_{args.gym_id}_{args.seed}_{int(time.time())}'
    root_dir = os.path.join("runs", run_name)
    if args.track:
        import wandb
        wandb.tensorboard.patch(root_logdir=root_dir)
        wandb.init(
            project=args.wandb_project_name, 
            entity=args.wandb_entity, 
            name=run_name,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(root_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{arg}|{getattr(args, arg)}|" for arg in vars(args)])),
    )
    
    seed_everything(args.seed)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # vectorized environment
    """env = gym.make(args.gym_id,  render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: x % 10 == 0)
    observation = env.reset()
    episodic_return = 0
    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        if done:
            print(f"Episodic return: {info['episode']['r'][0]}")
            observation = env.reset()
    env.close()"""
    
    def make_env(gym_id=args.gym_id, seed=args.seed, idx=0, capture_video=False, run_name=""):
        def _thunk():
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", step_trigger=lambda x: x % 1000 == 0)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return _thunk
    
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(gym_id=args.gym_id, seed=args.seed + i, idx=i, capture_video=args.capture_video, run_name=run_name) 
            for i in range(args.n_envs)
        ]
    )
    #only discrete action space is supported
    assert isinstance(envs.action_space, gym.spaces.Discrete) or isinstance(envs.action_space, gym.spaces.MultiDiscrete), "Only discrete action space is supported"
    print("Observation space:", envs.single_observation_space.shape)
    print("Action space:", envs.single_action_space.n)
    
    agent = Agent(envs).to(device)
    print(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    
    
    
    
    #observations = envs.reset()

"""    for _ in range(2000):
        actions = envs.action_space.sample()
        observations, rewards, terminations, truncations, infos = envs.step(actions)
        #print(infos)
        if infos:
            for item in infos["final_info"]:
                if "episode" in item.keys():
                    print(f"Episodic return: {item['episode']['r'][0]}")

    envs.close()"""