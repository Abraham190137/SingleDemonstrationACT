from asyncio import Condition
from re import S
from sre_constants import SUCCESS
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import std, tqdm
from einops import rearrange

from utils import load_data # data functions
from my_utils import my_load_data # dataset
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from my_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from my_sim_env import CustomEnv
from typing import List, Dict, Tuple, Any
import json
import test
from tqdm import tqdm
import scipy.stats as stats

import IPython
e = IPython.embed

# DATA_FOLDER: str = "random_start_pos"

# if os.name == 'nt':
#     DATASET_DIR: str = 'C:/Research/Transformers/SingleDemoACT/data_local' + DATA_FOLDER
# else:
#     DATASET_DIR:str = '/home/aigeorge/research/SingleDemoACT/data_local/' + DATA_FOLDER
# NUM_EPISODES: int = 50
# EPISODE_LEN: int = 50
# CAMERA_NAMES: List[str] = ['top']#, 'front', 'side-left']
# IS_SIM: bool = True

SUCCESS_CONDITIONED: bool = False

def main(args):
    set_seed(1)
    print('gpu node', args['gpu'], type(args['gpu']))
    if args['gpu'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
    # command line parameters
    is_eval = args['eval']
    save_dir = args['save_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    ckpt_dir: str = os.path.join(save_dir, 'checkpoints')
    dataset_dir: str = os.path.join(save_dir, 'data')

    # read the meta_data folder:
    with open(os.path.join(save_dir, 'meta_data.json'), 'r') as f:
        meta_data: Dict[str, Any] = json.load(f)
    task_name: str = meta_data['task_name']
    num_episodes: int = meta_data['num_episodes']
    episode_len: int = meta_data['episode_length']
    camera_names: List[str] = meta_data['camera_names']
    is_sim: bool = meta_data['is_sim']
    

    # fixed parameters
    state_dim = 4
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'SUCCESS_CONDITIONED': SUCCESS_CONDITIONED,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'SUCCESS_CONDITIONED': SUCCESS_CONDITIONED,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'transparent_arm': meta_data['transparent_arm'],
        'include_failures': meta_data['include_failures'],
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # load dataset
    if SUCCESS_CONDITIONED:
        train_dataloader, val_dataloader, stats, is_sim= my_load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)
    else:
        train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(obs, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def weighted_std(values:torch.Tensor, k):
    weights = np.exp(-k * np.arange(len(values)))
    weights = weights / weights.sum()
    weights = torch.from_numpy(weights).cuda().unsqueeze(dim=1)
    diffs = values - values.mean(dim=0, keepdim=True)
    return (diffs.pow(2) * weights).sum(dim=0).sqrt()


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    transparent_arm = config['transparent_arm']
    onscreen_cam = 'angle'
    stop_on_success = False

    if onscreen_render:
        import cv2
        cv2.namedWindow("plots", cv2.WINDOW_NORMAL)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # print(stats)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    env = CustomEnv(task_name, inject_noise=False, camera_names=camera_names, onscreen_render=onscreen_render, K_POS=20, transparent_arm=transparent_arm)
    env_max_reward = env.MAX_REWARD
    dt = env.DT

    print("temporal_agg:", temporal_agg)

    query_frequency = policy_config['num_queries']
    print('query_frequency', query_frequency)
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 2) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in tqdm(range(num_rollouts)):
        print(f'Rollout {rollout_id}')
        rollout_id += 0
        ### set task
        obs = env.reset()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            last_t_for_agg = 0
            reset_agg = False
            for t in range(max_timesteps):

                ### process previous timestep to get qpos and image_list
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['position'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        # If we are including failures (ie, conditioning it based on success), then we need to pass the success condition to the policy as part of the qpos.
                        # We assume that success is 1 (we want to generate succesful trajectories), and then concat
                        if SUCCESS_CONDITIONED:
                            # print('qpos', qpos.shape, qpos)
                            all_actions = policy(qpos, curr_image, is_success=torch.tensor([1]).float().to(qpos.device))
                        else:
                            all_actions = policy(qpos, curr_image)
                            # print('forward')

                    # Original
                    # if temporal_agg:
                    #     all_time_actions[[t], t:t+num_queries] = all_actions
                    #     actions_for_curr_step = all_time_actions[:, t]
                    #     actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    #     actions_for_curr_step = actions_for_curr_step[actions_populated]
                    #     k = 0.1
                    #     exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    #     exp_weights = exp_weights / exp_weights.sum()
                    #     exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    #     raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    # else:
                    #     raw_action = all_actions[:, t % query_frequency]

                    # Just K
                    # if temporal_agg:
                    #     # print('doing just k')
                    #     all_time_actions[[t], t:t+num_queries] = all_actions
                    #     actions_for_curr_step = all_time_actions[:, t]
                    #     actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    #     actions_for_curr_step = deepcopy(actions_for_curr_step[actions_populated])
                    #     if len(actions_for_curr_step) > 5:
                    #         # print(actions_for_curr_step)
                    #         std_adjust = torch.std(actions_for_curr_step, axis = 0)
                    #         # std_adjust = weighted_std(actions_for_curr_step, 0.1)
                    #         std_move = float(torch.max(std_adjust[:3]).cpu())
                    #         std_grip = float(std_adjust[3].cpu())
                    #         k_move = std_move/4
                    #         k_grip = std_grip/4
                    #         # print('K_move:', k_move, '\tK_grip', k_grip, '\tstd:', std_adjust)
                    #     else:
                    #         k_move = 0.1
                    #         k_grip = 0.1
                            

                    #     exp_weights_grip = np.exp(-k_grip * np.arange(len(actions_for_curr_step)))
                    #     exp_weights_grip = exp_weights_grip / exp_weights_grip.sum()
                    #     exp_weights_grip = torch.from_numpy(exp_weights_grip).cuda().unsqueeze(dim=1)

                    #     exp_weights_move = np.exp(-k_move * np.arange(len(actions_for_curr_step)))
                    #     exp_weights_move = exp_weights_move / exp_weights_move.sum()
                    #     exp_weights_move = torch.from_numpy(exp_weights_move).cuda().unsqueeze(dim=1)

                    #     raw_action = torch.empty(4).cuda()
                    #     raw_action[:3] = (actions_for_curr_step[:, :3] * exp_weights_move).sum(dim=0, keepdim=True)
                    #     raw_action[3] = (actions_for_curr_step[:, 3:] * exp_weights_grip).sum(dim=0, keepdim=True)
                        
                    # else:
                    #     raw_action = all_actions[:, t % query_frequency]
                    
                    # K and reset
                    if temporal_agg:
                        # print('k and reset')
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[last_t_for_agg:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = deepcopy(actions_for_curr_step[actions_populated])
                        if len(actions_for_curr_step) > 5 and not reset_agg:
                            # print(actions_for_curr_step)
                            std_adjust = torch.std(actions_for_curr_step, axis = 0)
                            # std_adjust = weighted_std(actions_for_curr_step, 0.1)
                            std_move = float(torch.max(std_adjust[:3]).cpu())
                            std_grip = float(std_adjust[3].cpu())
                            k_move = std_move
                            k_grip = std_grip
                            # print('K_move:', k_move, '\tK_grip', k_grip, '\tstd:', std_adjust)
                            if k_move > 0.5:
                                last_t_for_agg = t
                                reset_agg = True
                                reset_traj = all_actions
                                reset_t = t
                                # print('reset_traj', reset_traj)
                        else:
                            k_move = 0.1
                            k_grip = 0.1
                            
                        if reset_agg:
                            actions_for_curr_step = reset_traj[:, t-reset_t, :]
                            if t-reset_t > 25/2:
                                reset_agg = False
                            # print('reset action', actions_for_curr_step)


                        exp_weights_grip = np.exp(-k_grip * np.arange(len(actions_for_curr_step)))
                        exp_weights_grip = exp_weights_grip / exp_weights_grip.sum()
                        exp_weights_grip = torch.from_numpy(exp_weights_grip).cuda().unsqueeze(dim=1)

                        exp_weights_move = np.exp(-k_move * np.arange(len(actions_for_curr_step)))
                        exp_weights_move = exp_weights_move / exp_weights_move.sum()
                        exp_weights_move = torch.from_numpy(exp_weights_move).cuda().unsqueeze(dim=1)

                        raw_action = torch.empty(4).cuda()
                        raw_action[:3] = (actions_for_curr_step[:, :3] * exp_weights_move).sum(dim=0, keepdim=True)
                        raw_action[3] = (actions_for_curr_step[:, 3:] * exp_weights_grip).sum(dim=0, keepdim=True)

                        # k = 0.1
                        # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        # exp_weights = exp_weights / exp_weights.sum()
                        # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        # print(k)
                        
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    if SUCCESS_CONDITIONED:
                        raw_action = policy(qpos, curr_image, is_success=torch.tensor([1]).float().to(qpos.device))
                    else:
                        raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                # print('target', target_qpos)
                target_pose = env.unnormalize_grip(target_qpos)
                if False:
                    test.plot_action(env.env_dict["achieved_goal"][0:3], target_pose)
                    plt.savefig('temp.png')
                    cv2.imshow('plots',cv2.imread('temp.png'))
                obs = env.step_normalized_grip(target_qpos)
                # print('acheived', obs['position'])

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(obs['reward'])
                if obs['reward'] == env_max_reward and stop_on_success:
                    break

            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, dt, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    if SUCCESS_CONDITIONED:
        image_data, qpos_data, action_data, is_pad, is_success = data
        image_data, qpos_data, action_data, is_pad, is_success = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), is_success.cuda()
        return policy(qpos_data, image_data, action_data, is_pad, is_success = is_success)
    else:
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            # torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # save history:
    with open(os.path.join(ckpt_dir, 'train_history.pkl'), 'wb') as f:
        pickle.dump(train_history, f)
    with open(os.path.join(ckpt_dir, 'validation_history.pkl'), 'wb') as f:
        pickle.dump(validation_history, f)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    ### NOTE: Any changes to the parser must also be done in detr/main.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--save_dir', action='store', type=str, help='Directory where the checkpoint director will be created, where the recoreded sim episodes are saved, and where the meta-data JSON is located', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # For chosing gpu
    parser.add_argument('--gpu', action='store', type=int, help='chose which gpu to use', required=False)
    
    main(vars(parser.parse_args()))
