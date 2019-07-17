import os, sys, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def conv1d_output_size(h, k_h, s, p):
    h_out = int(((h - k_h + 2 * p) / s) + 1)
    return h_out


def conv2d_output_size(w, h, k_w, k_h, s, p):
    w_out = int(((w - k_w + 2 * p) / s) + 1)
    h_out = int(((h - k_h + 2 * p) / s) + 1)
    return w_out, h_out


def convtr1d_output_size(h, k_h, s, p):
    h_out = s * (h - 1) + k_h - 2 * p
    return h_out


def convtr2d_output_size(w, h, k_w, k_h, s, p):
    w_out = s * (w - 1) + k_w - 2 * p
    h_out = s * (h - 1) + k_h - 2 * p
    return w_out, h_out


def get_timestep(whole_data, step):
    return 120000


def get_expconfig(args):
    config = f'use_fc_{args.use_fc}_' \
        f'use_cols_{args.use_nonunique_cols}_' \
        f'norm_{args.normalize}_' \
        f'k_{args.kernels}_' \
        f'ch_{args.n_ch}_' \
        f'loss_{args.loss_func}_' \
        f'lr_{args.lr}'

    return config


def get_paths(args, exp_config):
    result_dir = os.path.join(args.result_dir,
                              f'recipe_{args.recipe}',
                              f'step_{args.step}',
                              f'learning_{args.learning_method}',
                              f'model_{args.model}',
                              exp_config)
    save_dir = os.path.join(args.save_dir,
                            f'recipe_{args.recipe}',
                            f'step_{args.step}',
                            f'learning_{args.learning_method}',
                            f'model_{args.model}',
                            exp_config)
    log_dir = os.path.join(args.log_dir,
                           f'recipe_{args.recipe}',
                           f'step_{args.step}',
                           f'learning_{args.learning_method}',
                           exp_config)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return result_dir, save_dir, log_dir


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('nn.Linear') != -1:
        nn.init.xavier_normal_(m.weight)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
        # m.weight.data.normal_(1.0, 0.02)


def get_actfn(key):
    actfn_dict = dict()
    actfn_dict['none'] = None
    actfn_dict['elu'] = nn.ELU()
    actfn_dict['tanh'] = nn.Tanh()
    actfn_dict['relu'] = nn.ReLU()
    actfn_dict['sigmoid'] = nn.Sigmoid()
    actfn_dict['leakyrelu'] = nn.LeakyReLU()
    if key not in actfn_dict.keys():
        raise NotImplementedError('Enter the proper input actfn')
    return actfn_dict[key]
