import argparse
import os
from distutils.util import strtobool
import gym
from torch.utils.tensorboard import SummaryWriter
import time
from lightning import seed_everything
import torch

def parse_args():
    parse = argparse.ArgumentParser()
    # environment setting
    parse.add_argument('--seed', type=int, default=1, help='random seed for reproducibility')
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

    return parse.parse_args()

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
    env = gym.make(args.gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: x % 100 == 0)
    observation = env.reset()
    episodic_return = 0
    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        if done:
            print(f"Episodic return: {info['episode']['r']}")
            observation = env.reset()
    env.close()