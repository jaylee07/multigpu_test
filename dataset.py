import copy
import torch
from torch.utils.data import Dataset

total_columns = ['Time', 'A Feed', 'D Feed',
                 'E Feed', 'A + C Feed', 'Recycle flow',
                 'Reactor feed', 'Reactor pressure', 'Reactor level',
                 'Reactor temperature', 'Purge rate', 'Seperator temperature',
                 'Seperator level', 'Seperator pressure', 'Seperator underflow',
                 'Stripper level', 'Stripper pressure', 'Stripper underflow',
                 'Stripper temperature', 'Stripper Steam flow_meas', 'Compressor work',
                 'Reactor cooling water temperature', 'Condensor cooling water temperature', 'Feed %A',
                 'Feed %B', 'Feed %C', 'Feed %D',
                 'Feed %E', 'Feed %F', 'Purge %A',
                 'Purge %B', 'Purge %C', 'Purge %D',
                 'Purge %E', 'Purge %F', 'Purge %G',
                 'Purge %H', 'Product %D', 'Product %E',
                 'Product %F', 'Product %G', 'Product %H',
                 'D feed flow', 'E feed flow', 'A feed flow',
                 'C feed flow', 'Compressor recycle valve', 'Purge flow',
                 'Separator liquid flow', 'Stripper liquid product flow', 'Stripper Steam flow_mv',
                 'Reactor cooling water flow', 'Condenser cooling water flow', 'Reactor Agitator speed',
                 'is_mv_attack', 'is_meas_attack', 'is_sp_attack',
                 'state', 'product_rate', 'hourly_cost']

measured_columns = ['A Feed', 'D Feed', 'E Feed',
                    'A + C Feed', 'Recycle flow', 'Reactor feed',
                    'Reactor pressure', 'Reactor level','Reactor temperature',
                    'Purge rate', 'Seperator temperature', 'Seperator level',
                    'Seperator pressure', 'Seperator underflow', 'Stripper level',
                    'Stripper pressure', 'Stripper underflow',  'Stripper temperature',
                    'Stripper Steam flow_meas', 'Compressor work', 'Reactor cooling water temperature',
                    'Condensor cooling water temperature', 'Feed %A', 'Feed %B',
                    'Feed %C', 'Feed %D', 'Feed %E',
                    'Feed %F', 'Purge %A', 'Purge %B',
                    'Purge %C', 'Purge %D', 'Purge %E',
                    'Purge %F', 'Purge %G', 'Purge %H',
                    'Product %D', 'Product %E', 'Product %F',
                    'Product %G', 'Product %H']

manipulated_columns = ['D feed flow', 'E feed flow', 'A feed flow',
                       'C feed flow', 'Compressor recycle valve', 'Purge flow',
                       'Separator liquid flow', 'Stripper liquid product flow', 'Stripper Steam flow_mv',
                       'Reactor cooling water flow', 'Condenser cooling water flow', 'Reactor Agitator speed']

attack_columns = ['is_mv_attack', 'is_meas_attack', 'is_sp_attack']
general_columns = ['Time', 'state', 'product_rate', 'hourly_cost']


class TWDataset(Dataset):
    def __init__(self, total_twlist, transform=None):
        self.total_twlist = total_twlist
        self.transform = transform

    def __len__(self):
        return len(self.total_twlist)

    def __getitem__(self, idx):
        data = self.total_twlist[idx]
        sample = {
            'x': data[measured_columns + manipulated_columns],
            'general': data[general_columns],
            'attack': data[attack_columns]
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class Preprocessing(object):
    def __init__(self, used_cols, stat_dict, normalize_method='minmax'):
        self.used_cols = used_cols
        self.stat_dict = stat_dict
        self.normalize_method = normalize_method

    def change_bool_to_float(self, df):
        df *= 1
        df = df.astype(float)
        return df

    def __call__(self, sample):
        eps = 1e-7
        x = sample['x'][self.used_cols]
        new_x = copy.copy(x)
        for col in self.used_cols:
            stats = self.stat_dict[col]
            if self.normalize_method == 'none':
                new_x[col] = new_x[col]
            elif self.normalize_method == 'z':
                new_x[col] = (new_x[col] - stats['mean']) / (stats['std'] + eps)
            elif self.normalize_method == 'minmax':
                new_x[col] = (new_x[col] - stats['min']) / (stats['max'] - stats['min'] + eps)
        new_sample = {
            'x': new_x.values,
            'general': sample['general'].values,
            'attack': self.change_bool_to_float(sample['attack']).values
        }

        return new_sample


class ToTensor(object):
    def __call__(self, sample):
        new_sample = {
            'x': torch.FloatTensor(sample['x']),
            'general': torch.FloatTensor(sample['general']),
            'attack': torch.FloatTensor(sample['attack'])
        }

        return new_sample