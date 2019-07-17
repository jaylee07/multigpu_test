import argparse
import preprocess
import dataset
import utils
import train
import ae

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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


def arguments():
    parser = argparse.ArgumentParser(description='multi-gpu experiment comparison')
    # data arguments
    parser.add_argument('--datadir', type=str, default='/data/jehyuk/TEP/')
    parser.add_argument('--trn_data_type', type=str, default='single', choices=['single', 'attack', 'transient'])
    parser.add_argument('--tst_data_type', type=str, default='attack', choices=['single', 'attack', 'transient'])
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--type', type=int, default=21, choices=[21, 22, 23, 24])
    parser.add_argument('--tw', type=int, default=500)
    # model arguments
    parser.add_argument('--k', type=int, nargs='+', default=[5, 5, 5, 5])
    parser.add_argument('--s', type=int, nargs='+', default=[1, 1, 1, 1])
    parser.add_argument('--p', type=int, nargs='+', default=[0, 0, 0, 0])
    parser.add_argument('--n_ch', type=int, default=128)
    parser.add_argument('--actfn_name', type=str, default='relu')
    parser.add_argument('--outactfn_name', type=str, default='sigmoid')
    parser.add_argument('--use_fc', default=False, action='store_true')
    parser.add_argument('--normalize', type=str, default='minmax', choices=['none', 'z', 'minmax'])
    # Train arguments
    parser.add_argument('--multigpu_method', type=int, default=1, choices=[1, 2])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--device_num', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--whole_devices', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--trn_batch_size', type=int, default=1024)
    parser.add_argument('--tst_batch_size', type=int, default=128)
    parser.add_argument('--loss_func', type=str, default='L1')
    parser.add_argument('--check_every', type=int, default=10)
    parser.add_argument('--result_dir', type=str, default='/home/jehyuk/multigpu_test/results')
    parser.add_argument('--save_dir', type=str, default='/home/jehyuk/multigpu_test/models')
    parser.add_argument('--log_dir', type=str, default='/home/jehyuk/multigpu_test/logs')
    parser.add_argument('--trn_mode', type=str, default='train', choices=['train', 'infer'])
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--load', default=False, action='store_true')

    args = parser.parse_args()

    return args

def main(args):
    trn_dir, trn_data_list = preprocess.get_datalist(args.datadir, args.mode, args.type, args.trn_data_type)
    # print(trn_dir)
    # print(trn_datalist)
    tst_dir, tst_data_list = preprocess.get_datalist(args.datadir, args.mode, args.type, args.tst_data_type)
    trn_twlist, df_total = preprocess.make_twlist(args.tw, trn_dir, trn_data_list, total_columns)
    stat_dict = preprocess.get_statdict(df_total, used_cols = measured_columns + manipulated_columns)
    del df_total
    tst_twlist, _ = preprocess.make_twlist(args.tw, tst_dir, tst_data_list, total_columns)

    transform_op = transforms.Compose([dataset.Preprocessing(used_cols=measured_columns + manipulated_columns,
                                                             stat_dict=stat_dict, normalize_method='minmax'),
                                       dataset.ToTensor()])
    trn_dset = dataset.TWDataset(trn_twlist, transform=transform_op)
    tst_dset = dataset.TWDataset(tst_twlist, transform=transform_op)
    trn_loader = DataLoader(trn_dset, batch_size=args.trn_batch_size, num_workers=args.n_workers, shuffle=True)
    tst_loader = DataLoader(tst_dset, batch_size=args.tst_batch_size, num_workers=args.n_workers, shuffle=False)

    model = ae.ConvAE(args.tw, measured_columns+manipulated_columns,
                      args.k, args.s, args.p, args.n_ch, args.use_fc, 20,
                      args.actfn_name, args.outactfn_name)
    device = torch.device(f'cuda:{args.device_num}')

    if args.multigpu_method == 1:
        trainer = train.AETrainer1(model, args.lr, args.weight_decay, device, args.whole_devices, args.check_every)
    elif args.multigpu_method == 2:
        trainer = train.AETrainer2(model, args.lr, args.weight_decay, device, args.whole_devices, args.check_every)

    trainer.train(trn_loader, args.n_epoch)
    trainer.save_model(args.save_dir)

args = arguments()
print('..')
main(args)