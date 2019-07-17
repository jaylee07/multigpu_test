import utils
import torch.nn as nn
from networks import EncoderConv1d, DecoderConv1d


class ConvAE(nn.Module):
    def __init__(self, tw, used_cols, k, s, p, n_ch, use_fc=False, fc_size=20,
                 actfn_name='relu', outactfn_name='sigmoid'):
        """
        :param tw: time window size
        :param used_cols: column list which is used in modeling
        :param k: kernel size list
        :param s: stride size list
        :param p: padding size list
        :param n_ch: channel bunch
        :param actfn_name: activation function in hidden layer. default='relu'
        :param outactfn_name: activation function in output layer. default='sigmoid'
        """
        super(ConvAE, self).__init__()
        self.tw = tw
        self.used_cols = used_cols
        self.k = k
        self.s = s
        self.p = p
        self.n_ch = n_ch
        embed_len = tw
        for i in range(len(k)):
            embed_len = utils.conv1d_output_size(embed_len, k[i], s[i], p[i])
        self.enc = EncoderConv1d(tw, len(used_cols), k, s, p, n_ch, actfn_name, use_fc, fc_size)
        self.dec = DecoderConv1d(tw, len(used_cols), k, s, p, n_ch, embed_len, actfn_name, outactfn_name, use_fc,
                                 fc_size)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat