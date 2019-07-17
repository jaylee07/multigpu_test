import os, sys, time, pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import utils
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

# single_datadir = '/data/jehyuk/TEP/single_states/'
# trans_datadir = '/data/jehyuk/TEP/transient_processes/'
# attack_datadir = '/data/jehyuk/TEP/attacks/'
#
# total_columns = ['Time', 'A Feed', 'D Feed',
#                  'E Feed', 'A + C Feed', 'Recycle flow',
#                  'Reactor feed', 'Reactor pressure', 'Reactor level',
#                  'Reactor temperature', 'Purge rate', 'Seperator temperature',
#                  'Seperator level', 'Seperator pressure', 'Seperator underflow',
#                  'Stripper level', 'Stripper pressure', 'Stripper underflow',
#                  'Stripper temperature', 'Stripper Steam flow_meas', 'Compressor work',
#                  'Reactor cooling water temperature', 'Condensor cooling water temperature', 'Feed %A',
#                  'Feed %B', 'Feed %C', 'Feed %D',
#                  'Feed %E', 'Feed %F', 'Purge %A',
#                  'Purge %B', 'Purge %C', 'Purge %D',
#                  'Purge %E', 'Purge %F', 'Purge %G',
#                  'Purge %H', 'Product %D', 'Product %E',
#                  'Product %F', 'Product %G', 'Product %H',
#                  'D feed flow', 'E feed flow', 'A feed flow',
#                  'C feed flow', 'Compressor recycle valve', 'Purge flow',
#                  'Separator liquid flow', 'Stripper liquid product flow', 'Stripper Steam flow_mv',
#                  'Reactor cooling water flow', 'Condenser cooling water flow', 'Reactor Agitator speed',
#                  'is_mv_attack', 'is_meas_attack', 'is_sp_attack',
#                  'state', 'product_rate', 'hourly_cost']
#
# measured_columns = ['A Feed', 'D Feed', 'E Feed',
#                     'A + C Feed', 'Recycle flow', 'Reactor feed',
#                     'Reactor pressure', 'Reactor level','Reactor temperature',
#                     'Purge rate', 'Seperator temperature', 'Seperator level',
#                     'Seperator pressure', 'Seperator underflow', 'Stripper level',
#                     'Stripper pressure', 'Stripper underflow',  'Stripper temperature',
#                     'Stripper Steam flow_meas', 'Compressor work', 'Reactor cooling water temperature',
#                     'Condensor cooling water temperature', 'Feed %A', 'Feed %B',
#                     'Feed %C', 'Feed %D', 'Feed %E',
#                     'Feed %F', 'Purge %A', 'Purge %B',
#                     'Purge %C', 'Purge %D', 'Purge %E',
#                     'Purge %F', 'Purge %G', 'Purge %H',
#                     'Product %D', 'Product %E', 'Product %F',
#                     'Product %G', 'Product %H']
#
# manipulated_columns = ['D feed flow', 'E feed flow', 'A feed flow',
#                        'C feed flow', 'Compressor recycle valve', 'Purge flow',
#                        'Separator liquid flow', 'Stripper liquid product flow', 'Stripper Steam flow_mv',
#                        'Reactor cooling water flow', 'Condenser cooling water flow', 'Reactor Agitator speed']
#
# attack_columns = ['is_mv_attack', 'is_meas_attack', 'is_sp_attack']
# general_columns = ['Time', 'state', 'product_rate', 'hourly_cost']

def get_datalist(data_dir, mode, type, data_type='single'):

    if data_type == 'attack':
        data_dir = os.path.join(data_dir, 'attacks')
        data_list = sorted(os.listdir(data_dir))
        datalist = sorted([x for x in data_list if f'mode_{mode}_type_{type}' in x])

    elif data_type == 'single':
        data_dir = os.path.join(data_dir, 'single_states')
        data_list = sorted(os.listdir(data_dir))
        datalist = sorted([x for x in data_list if f'mode_{mode}' in x])

    elif data_type == 'transient':
        data_dir = os.path.join(data_dir, 'transient_processes')
        data_list = sorted(os.listdir(data_dir))
        datalist = sorted([x for x in data_list if f'mode_{mode}' and '.csv' in x])

    return data_dir, datalist


def make_twlist(tw, datadir, datalist, total_columns):
    total_twlist = []
    total_dflist = []
    print(">> Making twlist..")
    for i, fname in tqdm.tqdm(enumerate(datalist), total=len(datalist)):
        df = pd.read_csv(os.path.join(datadir, fname), names=total_columns)
        twlist = [df[j-tw:j] for j in range(tw, df.shape[0])]
        total_dflist.append(df)
        total_twlist.extend(twlist)
    df_total =pd.concat([x for x in total_dflist], axis=0)
    return total_twlist, df_total


def get_statdict(df, used_cols):
    stat_dict = dict()
    for col in used_cols:
        stat_dict[col] = dict()
        stat_dict[col]['mean'] = df[col].mean()
        stat_dict[col]['std'] = df[col].std()
        stat_dict[col]['min'] = df[col].min()
        stat_dict[col]['max'] = df[col].max()
    return stat_dict