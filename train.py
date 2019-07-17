import os, pickle
import tqdm
import parallel

import torch
import torch.nn as nn
import torch.optim as optim


# Use nn.DataParallel
class AETrainer1:
    def __init__(self, model, lr, weight_decay, device, whole_devices, check_every=10):
        """
        :param model: AE model to train
        :param trn_loader: train data loader
        :param tst_loader: test data loader
        :param lr: learning rate
        :param weight_decay: weight decay
        :param device: torch.device('cuda:{}') where the main model is positioned
        :param whole_device: list of cuda numbers where the parallel operation is done
        :param check_every: logging frequency
        """
        self.model = model.to(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.whole_devices = whole_devices
        self.check_every = check_every
        if device.type == 'cuda' and torch.cuda.device_count()>1 and len(whole_devices) > 1:
            print("Using {} gpus for training AE".format(len(whole_devices)))
            self.model = nn.DataParallel(self.model, device_ids=whole_devices)
        self.optim = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        self.criterion = nn.MSELoss(reduction='sum')
        self.check_every = check_every

    def partial_fit(self, data_loader, epoch, train=True):
        """
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value whether it is train or test
        """
        str_code = 'train' if train else 'test'
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc='epoch_{}:{}'.format(str_code, epoch),
                              total=len(data_loader))
        avg_loss = 0
        for iter_num, batch in data_iter:
            if train:
                self.model.train()
            else:
                self.model.eval()
            self.optim.zero_grad()
            x=torch.transpose(batch['x'], 1, 2).to(self.device)
            general = batch['general'].to(self.device)
            attack = batch['attack'].to(self.device)
            x_hat = self.model(x)
            loss = self.criterion(x_hat, x)
            avg_loss += loss.item()
            loss.backward()
            self.optim.step()
            post_fix = {
                "epoch": epoch,
                "iter": iter_num+1,
                "avg_loss": avg_loss / (data_loader.batch_size*(iter_num+1)),
                "batch_loss": loss.item() / data_loader.batch_size
            }
            if (iter_num+1) % self.check_every == 0:
                data_iter.write(str(post_fix))

    def train(self, data_loader, n_epoch, train=True):
        for epoch in tqdm.tqdm(range(1, n_epoch+1)):
            self.partial_fit(data_loader, epoch, train)

    def test(self, data_loader, epoch=0, train=False):
        self.partial_fit(data_loader, epoch, train)

    def save_model(self, save_dir):
        if self.device.type == 'cuda' and torch.cuda.device_count()>1 and len(self.whole_devices)>1:
            torch.save(self.model.module.encoder.state_dict(), os.path.join(save_dir, 'encoder.pkl'))
            torch.save(self.model.module.decoder.state_dict(), os.path.join(save_dir, 'decoder.pkl'))
        else:
            torch.save(self.model.encoder.state_dict(), os.path.join(save_dir, 'encoder.pkl'))
            torch.save(self.model.decoder.state_dict(), os.path.join(save_dir, 'decoder.pkl'))

    def load_model(self, save_dir):
        if self.device.type == 'cuda' and torch.cuda.device_count()>1 and len(self.whole_devices)>1:
            self.model.module.encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pkl')))
            self.model.module.decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pkl')))
        else:
            self.model.encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pkl')))
            self.model.decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pkl')))


# Use Customized Parallel
class AETrainer2:
    def __init__(self, model, lr, weight_decay, device, whole_devices, check_every=10):
        """
        :param model: AE model to train
        :param trn_loader: train data loader
        :param tst_loader: test data loader
        :param lr: learning rate
        :param weight_decay: weight decay
        :param device: torch.device('cuda:{}') where the main model is positioned
        :param whole_device: list of cuda numbers where the parallel operation is done
        :param check_every: logging frequency
        """
        torch.cuda.empty_cache()
        self.model = model.to(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.whole_devices = whole_devices
        self.check_every = check_every
        self.criterion = nn.MSELoss(reduction='sum')
        self.optim = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        if device.type == 'cuda' and torch.cuda.device_count() > 1 and len(whole_devices) > 1:
            print("Using {} gpus for training AE".format(len(whole_devices)))
            self.model = parallel.DataParallelModel(self.model).to(device)
            self.criterion = parallel.DataParallelCriterion(self.criterion).to(device)
        self.check_every = check_every

    def partial_fit(self, data_loader, epoch, train=True):
        """
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value whether it is train or test
        """
        str_code = 'train' if train else 'test'
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc='epoch_{}:{}'.format(str_code, epoch),
                              total=min(len(data_loader), 100))
        avg_loss = 0
        for iter_num, batch in data_iter:
            if train:
                self.model.train()
            else:
                self.model.eval()
            self.optim.zero_grad()
            x = torch.transpose(batch['x'], 1, 2).to(self.device)
            general = batch['general'].to(self.device)
            attack = batch['attack'].to(self.device)
            x_hat = self.model(x)
            loss = self.criterion(x_hat, x)
            avg_loss += loss.item()
            loss.backward()
            self.optim.step()
            post_fix = {
                "epoch": epoch,
                "iter": iter_num + 1,
                "avg_loss": avg_loss / (data_loader.batch_size * (iter_num + 1)),
                "batch_loss": loss.item() / data_loader.batch_size
            }
            if (iter_num + 1) % self.check_every == 0:
                data_iter.write(str(post_fix))

            if (iter_num + 1) == 50:
                break

    def train(self, data_loader, n_epoch, train=True):
        for epoch in tqdm.tqdm(range(1, n_epoch + 1)):
            self.partial_fit(data_loader, epoch, train)

    def test(self, data_loader, epoch=0, train=False):
        self.partial_fit(data_loader, epoch, train)

    def save_model(self, save_dir):
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1 and len(self.whole_devices) > 1:
            torch.save(self.model.module.encoder.state_dict(), os.path.join(save_dir, 'encoder.pkl'))
            torch.save(self.model.module.decoder.state_dict(), os.path.join(save_dir, 'decoder.pkl'))
        else:
            torch.save(self.model.encoder.state_dict(), os.path.join(save_dir, 'encoder.pkl'))
            torch.save(self.model.decoder.state_dict(), os.path.join(save_dir, 'decoder.pkl'))

    def load_model(self, save_dir):
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1 and len(self.whole_devices) > 1:
            self.model.module.encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pkl')))
            self.model.module.decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pkl')))
        else:
            self.model.encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pkl')))
            self.model.decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pkl')))