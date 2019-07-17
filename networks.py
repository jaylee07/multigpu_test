from collections import OrderedDict
import utils
import torch
import torch.nn as nn


class EncoderConv1d(nn.Module):
    def __init__(self, tw, n_vars, kernels, strides, paddings, n_ch,
                 actfn_name='relu', use_fc=False, fc_size=20):
        super(EncoderConv1d, self).__init__()
        self.tw = tw
        self.n_vars = n_vars
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.use_fc = use_fc
        self.n_ch =n_ch

        k, s, p = kernels, strides, paddings
        length = tw
        act_fn = utils.get_actfn(actfn_name)

        layers = OrderedDict()
        in_ch, out_ch = n_vars, self.n_ch
        for i in range(len(k)-1):
            layers[f'conv{i+1}'] = nn.Conv1d(in_ch, out_ch, k[i], s[i], p[i], bias=False)
            layers[f'bn{i+1}'] = nn.BatchNorm1d(num_features=out_ch)
            layers[f'act{i+1}'] = act_fn
            length = utils.conv1d_output_size(length, k[i], s[i], p[i])
            in_ch, out_ch = out_ch, out_ch*2
        i += 1
        layers[f'conv{i+1}'] = nn.Conv1d(in_ch, out_ch, k[i], s[i], p[i], bias=True) # No batchnorm -> Bias!
        length = utils.conv1d_output_size(length, k[i], s[i], p[i])
        self.layers = nn.Sequential(layers)
        if self.use_fc:
            self.fc = nn.Linear(length * out_ch, fc_size)

    def forward(self, x):
        out = self.layers(x)
        if self.use_fc:
            out = torch.flatten(out, start_dim=1)
            out = self.fc(out)
        return out


class DecoderConv1d(nn.Module):
    def __init__(self, tw, n_vars, kernels, strides, paddings, n_ch,
                 embed_length, actfn_name='relu', outactfn_name='sigmoid', use_fc=False, fc_size=20):
        super(DecoderConv1d, self).__init__()
        self.tw = tw
        self.n_vars = n_vars
        self.kernels = [x for x in reversed(kernels)]
        self.strides = [x for x in reversed(strides)]
        self.paddings = [x for x in reversed(paddings)]
        self.n_ch = n_ch
        self.embed_length = embed_length
        self.use_fc = use_fc
        self.fc_size = fc_size

        k, s, p = self.kernels, self.strides, self.paddings

        in_ch = self.n_ch * (2 ** (len(k) - 1))
        out_ch = int(in_ch / 2)
        if self.use_fc:
            self.fc = nn.Linear(self.fc_size, embed_length * in_ch)
        length = embed_length
        act_fn = utils.get_actfn(actfn_name)
        outact_fn = utils.get_actfn(outactfn_name)
        layers = OrderedDict()
        for i in range(len(k) - 1):
            layers[f'convtr{i + 1}'] = nn.ConvTranspose1d(in_ch, out_ch, k[i], s[i], p[i], bias=False)
            layers[f'bn{i + 1}'] = nn.BatchNorm1d(num_features=out_ch)
            layers[f'act{i + 1}'] = act_fn
            length = utils.convtr1d_output_size(length, k[i], s[i], p[i])
            in_ch, out_ch = out_ch, int(out_ch / 2)
        i += 1
        layers[f'convtr{i + 1}'] = nn.ConvTranspose1d(in_ch, n_vars, k[i], s[i], p[i], bias=True)
        # length = utils.convtr1d_output_size(length, k[i], s[i], p[i])
        self.layers = nn.Sequential(layers)
        self.outact_fn = outact_fn

    def forward(self, z):
        if self.use_fc:
            z = self.fc(z)
            out_ch = self.n_ch * (2 ** (len(self.kernels) - 1))
            z = z.view(-1, out_ch, self.embed_length)
        out = self.layers(z)
        if self.outact_fn is not None:
            out = self.outact_fn(out)
        return out

