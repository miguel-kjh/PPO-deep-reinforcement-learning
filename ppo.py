import argparse
import os
from distutils.util import strtobool
import numpy as np
import random
import torch

import gym

import time
from torch.utils.tensorboard import SummaryWriter

# delete warnings
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    args = parser.parse_args()
    return args

def make_env(gym_id):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordVideo(
            env, 
            f"videos", 
            episode_trigger = lambda x: x % 100 == 0,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def main():
    args = parse_args()
    run_name = f"{args.gym_id}_{args.seed}_{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True
        )

    writer = SummaryWriter(f"runs/{args.exp_name}/{run_name}")
    writer.add_text(
        "hyperparameters", 
        f"total_timesteps: {args.total_timesteps}, lr: {args.learning_rate}"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    observation = envs.reset()
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, done, truncated, info = envs.step(action)
        if done or truncated:
            print(f"reward: {info['final_info'][0]['episode']['r']}") 
    

if __name__ == "__main__":
    main()