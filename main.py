from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools

from scores.eval import generate_scores, evaluate_scores

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-d', '--dataset', type=str, required=False, help="path to h5 dataset (required)")
parser.add_argument('-us', '--userscore', type=str, required=True, help="path to h5 of user's scores (required)")
parser.add_argument('-s', '--split', type=str, required=True, help="path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'],
                    help="evaluation metric ['tvsum', 'summe']")
# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=512, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=2, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='gru', help="RNN cell type (default: lstm)")
# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=5, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

regularization_factor = 0.15

def reconstruction_loss(h_origin, h_sum):
    """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

    return torch.norm(h_origin - h_sum, p=2)

def sparsity_loss(scores):
    """Summary-Length Regularization"""

    return torch.abs(torch.mean(scores) - regularization_factor)

def main():
    
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    if args.dataset is None:
        datasets = ['datasets/eccv16_dataset_summe_google_pool5.h5',
                'datasets/eccv16_dataset_tvsum_google_pool5.h5',
                'datasets/eccv16_dataset_ovp_google_pool5.h5',
                'datasets/eccv16_dataset_youtube_google_pool5.h5']
        
        dataset = {}
        for name in datasets:
            _, base_filename = os.path.split(name)
            base_filename = os.path.splitext(base_filename)
            dataset[base_filename[0]] = h5py.File(name, 'r')
        # Load split file
        splits = read_json(args.split)
        assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
        split = splits[args.split_id]
        train_keys = split['train_keys']
        test_keys = split['test_keys']
        print("# train videos {}. # test videos {}".format(len(train_keys), len(test_keys)))
    
    else:
        dataset = h5py.File(args.dataset, 'r')
        num_videos = len(dataset.keys())
        splits = read_json(args.split)
        assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
        split = splits[args.split_id]
        train_keys = split['train_keys']
        test_keys = split['test_keys']
        print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

    #### Set User Score Dataset ####
    userscoreset = h5py.File(args.userscore, 'r')

    print("Initialize model")
    model = DSRRL(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.5,0.999) ,lr=args.lr, weight_decay=args.weight_decay)
    
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        evaluate(model, dataset, test_keys, use_gpu)
        return

    if args.dataset is None:
        print("==> Start training")
        start_time = time.time()
        model.train()
        baselines = {key: 0. for key in train_keys} # baseline rewards for videos
        reward_writers = {key: [] for key in train_keys} # record reward changes for each video

        for epoch in range(start_epoch, args.max_epoch):
            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs) # shuffle indices

            for idx in idxs:
                key_parts = train_keys[idx].split('/')
                name, key = key_parts
                seq = dataset[name][key]['features'][...] # sequence of features, (seq_len, dim)
                seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
                if use_gpu: seq = seq.cuda()
                probs, out_feats, att_score = model(seq) # output shape (1, seq_len, 1)

                cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term
                m = Bernoulli(probs)
                epis_rewards = []
                for _ in range(args.num_episode):
                    actions = m.sample()
                    log_probs = m.log_prob(actions)
                    reward = compute_reward(seq, actions, use_gpu=use_gpu)
                    expected_reward = log_probs.mean() * (reward - baselines[train_keys[idx]])
                    cost -= expected_reward
                    epis_rewards.append(reward.item())
                
                recon_loss = reconstruction_loss(seq, out_feats)
                spar_loss = sparsity_loss(att_score)

                total_loss = cost + recon_loss + spar_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                baselines[train_keys[idx]] = 0.9 * baselines[train_keys[idx]] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
                reward_writers[train_keys[idx]].append(np.mean(epis_rewards))

            epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
            #print("epoch {}/{}\t reward {}\t loss {}".format(epoch+1, args.max_epoch, epoch_reward, total_loss)) 
    else:
        print("==> Start training")
        start_time = time.time()
        model.train()
        baselines = {key: 0. for key in train_keys} # baseline rewards for videos
        reward_writers = {key: [] for key in train_keys} # record reward changes for each video

        for epoch in range(start_epoch, args.max_epoch):
            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs) # shuffle indices

            for idx in idxs:
                key = train_keys[idx]
                seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
                seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
                if use_gpu: seq = seq.cuda()
                probs, out_feats, att_score = model(seq) # output shape (1, seq_len, 1)

                cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term
                m = Bernoulli(probs)
                epis_rewards = []
                for _ in range(args.num_episode):
                    actions = m.sample()
                    log_probs = m.log_prob(actions)
                    reward = compute_reward(seq, actions, use_gpu=use_gpu)
                    expected_reward = log_probs.mean() * (reward - baselines[key])
                    cost -= expected_reward
                    epis_rewards.append(reward.item())

                recon_loss = reconstruction_loss(seq, out_feats)
                spar_loss = sparsity_loss(att_score)

                total_loss = cost + recon_loss + spar_loss

                #print(cost.item(), recon_loss.item(), spar_loss.item())

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
                reward_writers[key].append(np.mean(epis_rewards))

            epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
            #print("epoch {}/{}\t reward {}\t loss {}".format(epoch+1, args.max_epoch, epoch_reward, total_loss))
        
    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))
    evaluate(model, dataset, userscoreset, test_keys, use_gpu)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
    model_save_path = osp.join(args.save_dir, args.metric+'_model_epoch_' + str(args.max_epoch) +'_split_id_' + str(args.split_id) + '-' + str(args.rnn_cell) + '.pth.tar')
    save_checkpoint(model_state_dict, model_save_path)
    print("Model saved to {}".format(model_save_path))

def evaluate(model, dataset, userscoreset, test_keys, use_gpu):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'
        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result_ep{}_split_{}_{}.h5'.format(args.max_epoch, args.split_id, args.rnn_cell)), 'w')
        
        spear_avg_corrs = []
        kendal_avg_corrs = []

        if args.dataset is None:
            for key_idx, _ in enumerate(test_keys):
                key_parts = test_keys[key_idx].split('/')
                name, key = key_parts
                seq = dataset[name][key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                if use_gpu: seq = seq.cuda()
                probs, _, _ = model(seq)
                probs = probs.data.cpu().squeeze().numpy()
                cps = dataset[name][key]['change_points'][...]
                num_frames = dataset[name][key]['n_frames'][()]
                nfps = dataset[name][key]['n_frame_per_seg'][...].tolist()
                positions = dataset[name][key]['picks'][...]
                user_summary = dataset[name][key]['user_summary'][...]

                gtscore = dataset[name][key]['gtscore'][...]

                machine_summary, gt_frame_score = vsum_tools.generate_summary(probs, gtscore, cps, num_frames, nfps, positions)
                fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
                fms.append(fm)

                #### Calculate correlation matrices ####
                user_scores = userscoreset[key]["user_scores"][...]
                machine_scores = generate_scores(probs, num_frames, positions)
                spear_avg_corr = evaluate_scores(machine_scores, user_scores, metric="spearmanr")
                kendal_avg_corr = evaluate_scores(machine_scores, user_scores, metric="kendalltau")

                spear_avg_corrs.append(spear_avg_corr)
                kendal_avg_corrs.append(kendal_avg_corr)

                if args.verbose:
                    table.append([key_idx+1, key, "{:.1%}".format(fm)])

                if args.save_results:
                    h5_res.create_dataset(key + '/gt_frame_score', data=gt_frame_score)
                    h5_res.create_dataset(key + '/score', data=probs)
                    h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                    h5_res.create_dataset(key + '/gtscore', data=dataset[name][key]['gtscore'][...])
                    h5_res.create_dataset(key + '/fm', data=fm)
        else:
            for key_idx, key in enumerate(test_keys):
                seq = dataset[key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                if use_gpu: seq = seq.cuda()
                probs, _, _ = model(seq)
                probs = probs.data.cpu().squeeze().numpy()
                cps = dataset[key]['change_points'][...]
                num_frames = dataset[key]['n_frames'][()]
                nfps = dataset[key]['n_frame_per_seg'][...].tolist()
                positions = dataset[key]['picks'][...]
                user_summary = dataset[key]['user_summary'][...]

                gtscore = dataset[key]['gtscore'][...]

                machine_summary, gt_frame_score = vsum_tools.generate_summary(probs, gtscore, cps, num_frames, nfps, positions)
                fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
                fms.append(fm)

                #### Calculate correlation matrices ####
                user_scores = userscoreset[key]["user_scores"][...]
                machine_scores = generate_scores(probs, num_frames, positions)
                spear_avg_corr = evaluate_scores(machine_scores, user_scores, metric="spearmanr")
                kendal_avg_corr = evaluate_scores(machine_scores, user_scores, metric="kendalltau")

                spear_avg_corrs.append(spear_avg_corr)
                kendal_avg_corrs.append(kendal_avg_corr)

                if args.verbose:
                    table.append([key_idx+1, key, "{:.1%}".format(fm)])

                if args.save_results:
                    h5_res.create_dataset(key + '/gt_frame_score', data=gt_frame_score)
                    h5_res.create_dataset(key + '/score', data=probs)
                    h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                    h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                    h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F1-score {:.1%}".format(mean_fm))

    mean_spear_avg = np.mean(spear_avg_corrs)
    mean_kendal_avg = np.mean(kendal_avg_corrs)
    print("Average Kendal {}".format(mean_kendal_avg))
    print("Average Spear {}".format(mean_spear_avg))

    return mean_fm

if __name__ == '__main__':
    main()
