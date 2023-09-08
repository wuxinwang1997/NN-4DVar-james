import torch
import torch.nn as nn
from src.utils.model_utils import setup_conv1d

class TinyNetwork(nn.Module):
    def __init__(self, kernel_size, n_filters_ks4, filters_ks1_init=None, filters_ks1_inter=None,
                 filters_ks1_final=None, n_channels_in=1, n_channels_out=1, padding_mode='zeros'):
        super(TinyNetwork, self).__init__()
        self.kernel_size = kernel_size
        self.n_filters_ks1 = [filters_ks1_inter for i in range(len(n_filters_ks4) - 1)]
        self.n_filters_ks1 = self.n_filters_ks1 + [filters_ks1_final]
        if filters_ks1_init is None:
            self.n_filters_ks1 = [[]] + self.n_filters_ks1
        else:
            self.n_filters_ks1 = [filters_ks1_init] + self.n_filters_ks1
        assert len(self.n_filters_ks1) == len(n_filters_ks4) + 1

        self.layers4x4 = []
        self.layers_ks1 = [[] for i in range(len(self.n_filters_ks1))]
        n_in = n_channels_in + 1
        for i in range(len(self.n_filters_ks1)):
            for j in range(len(self.n_filters_ks1[i])):
                n_out = self.n_filters_ks1[i][j]
                layer = setup_conv1d(in_channels=n_in,
                                   out_channels=n_out,
                                   kernel_size=self.kernel_size,
                                   bias=True,
                                   padding_mode=padding_mode)
                self.layers_ks1[i].append(layer)
                n_in = n_out

            if i >= len(n_filters_ks4):
                break

            n_out = n_filters_ks4[i]
            layer = setup_conv1d(in_channels=n_in,
                               out_channels=n_out,
                               kernel_size=self.kernel_size,
                               bias=True,
                               padding_mode=padding_mode)
            self.layers4x4.append(layer)
            n_in = n_out

        self.layers4x4 = torch.nn.ModuleList(self.layers4x4)
        self.layers1x1 = sum(self.layers_ks1, [])
        self.layers1x1 = torch.nn.ModuleList(self.layers1x1)
        self.final = torch.nn.Conv1d(in_channels=n_in,
                                     out_channels=n_channels_out,
                                     kernel_size=1)
        self.nonlinearity = torch.nn.ReLU()

    def forward(self, xb, obs):
        x = torch.concat([xb, obs], dim=1)
        for layer in self.layers_ks1[0]:
            x = self.nonlinearity(layer(x))
        for i, layer4x4 in enumerate(self.layers4x4):
            x = self.nonlinearity(layer4x4(x))
            for layer in self.layers_ks1[i + 1]:
                x = self.nonlinearity(layer(x))

        return self.final(x)


class TinyResNet(TinyNetwork):
    def forward(self, xb, obs):
        out = xb
        x = torch.concat([xb, obs], dim=1)
        #assert n_channels_in//2 == n_channels_in/2
        #out = x[:, n_channels_in//2:]

        for layer in self.layers_ks1[0]:
            x = self.nonlinearity(layer(x))

        for i, layer4x4 in enumerate(self.layers4x4):
            x = self.nonlinearity(layer4x4(x))
            for layer in self.layers_ks1[i+1]:
                x = self.nonlinearity(layer(x))

        return self.final(x) + out