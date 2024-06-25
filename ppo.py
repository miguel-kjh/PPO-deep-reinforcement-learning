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
    parse.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, help="whether to anneal the learning rate")
    parse.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, help="whether to use generalized advantage estimation")
    parse.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parse.add_argument("--gae-lambda", type=float, default=0.95, help="lambda for the GAE")
    parse.add_argument("--num_minibatches", type=int, default=4, help="the number of minibatches")
    parse.add_argument("--update-epochs", type=int, default=4, help="the number of epochs to update the network")
    parse.add_argument("--norm-advs", type=lambda x: bool(strtobool(x)), default=True, help="whether to normalize the advantages")

    #clipping setting
    parse.add_argument("--clip-coef", type=float, default=0.2, help="the clipping loss")
    parse.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, help="whether to clip the value loss")
    #entropy setting
    parse.add_argument("--ent-coef", type=float, default=0.01, help="the entropy loss coefficient")
    parse.add_argument("--vf-coef", type=float, default=0.5, help="the value loss coefficient")
    parse.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    #early stopping setting
    parse.add_argument("--target_kl", type=float, default=None, help="the target kl divergence")

    args = parse.parse_args()
    args.batch_size = int(args.n_envs * args.n_steps)
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args

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
        
    def get_value(self, x):
        return self.critc(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)

if __name__ == "__main__":
    args = parse_args()
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
    
    agent = Agent(envs).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    #ALGO logic 
    obs = torch.zeros((args.n_steps, args.n_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long).to(device)
    log_probs = torch.zeros((args.n_steps, args.n_envs)).to(device)
    rewards = torch.zeros((args.n_steps, args.n_envs)).to(device)
    dones = torch.zeros((args.n_steps, args.n_envs)).to(device)
    values = torch.zeros((args.n_steps, args.n_envs)).to(device)
    
    #start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.n_envs).to(device)
    num_updates = args.total_timesteps // args.n_steps // args.n_envs
    
    
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = args.learning_rate * frac
            optimizer.param_groups[0]["lr"] = lr_now
            
        for step in range(args.n_steps):
            global_step += args.n_envs
            obs[step] = next_obs
            dones[step] = next_done
            
        with torch.no_grad():
            action, log_prob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob
        
        next_obs, reward, next_done, _, infos = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
        if infos:
            for item in infos["final_info"]:
                if item is not None and "episode" in item.keys():
                    print(f"Episodic return: {item['episode']['r'][0]}, global_step: {global_step}")
                    writer.add_scalar("charts/episode_return", item["episode"]["r"][0], global_step)
                    writer.add_scalar("charts/episode_length", item["episode"]["l"][0], global_step)
                    if args.track:
                        wandb.log({"charts/episode_return": item["episode"]["r"][0], "charts/episode_length": item["episode"]["l"][0], "global_step": global_step})
                    break
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.n_steps)):
                    if t == args.n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                # normal advantage
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.n_steps)):
                    if t == args.n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clip_fracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # get action and values
                _, log_prob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                log_ratio = log_prob - b_logprobs[mb_inds]
                ratio = torch.exp(log_ratio)

                #debug variables
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_advs:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                #policy loss
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                policy_loss  = torch.max(policy_loss1, policy_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # entropy loss
                entropy_loss = entropy.mean()

                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # early stopping
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        #other debug info
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track:
            wandb.log({
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": policy_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clip_fracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=global_step)


    # close environments to prevent memory leaks
    envs.close()
    if args.track:
        wandb.finish()
    writer.close()
    
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