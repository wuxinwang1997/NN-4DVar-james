import torch
import torch.nn as nn
from src.utils.model_utils import setup_conv1d

class ResNetBlock(nn.Module):
    """A residual block to construct residual networks.
    Comprises 2 conv1D operations with optional dropout and a normalization layer.

    Parameters
    ----------
    in_channels: int
        Number of channels of input tensor.
    kernel_size: list of (int, int)
        Size of the convolutional kernel for the residual layers.
    hidden_channels: int
        Number of output channels for first residual convolution.
    out_channels: int
        Number of output channels. If not equal to in_channels, will add
        additional 1x1 convolution.
    bias: bool
        Whether to include bias parameters in the residual-layer convolutions.
    layerNorm: function
        Normalization layer.
    activation: str
        String specifying nonlinearity.
    padding_mode: str
        How to pad the data ('circular' for wrap-around padding on last axis)
    dropout: float
        Dropout rate.
    """
    def __init__(self, in_channels, kernel_size,
                 hidden_channels=None, out_channels=None, additive=None,
                 bias=True, layerNorm=torch.nn.BatchNorm1d,
                 padding_mode='circular', dropout=0.1, activation="relu"):

        super(ResNetBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.additive = (out_channels == in_channels) if additive is None else additive

        self.conv1 = setup_conv1d(in_channels=in_channels,
                                  out_channels=hidden_channels,
                                  kernel_size=kernel_size,
                                  bias=bias,
                                  padding_mode=padding_mode)

        n_out_conv2 = out_channels if self.additive else hidden_channels
        self.conv2 = setup_conv1d(in_channels=hidden_channels,
                                  out_channels=n_out_conv2,
                                  kernel_size=kernel_size,
                                  bias=bias,
                                  padding_mode=padding_mode)

        if layerNorm is torch.nn.BatchNorm1d:
            self.norm1 = layerNorm(num_features=hidden_channels)
            self.norm2 = layerNorm(num_features=n_out_conv2)
        elif isinstance(layerNorm, torch.nn.Identity):
            self.norm1 = self.norm2 = layerNorm
        else:
            raise NotImplementedError

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        if not self.additive:
            self.conv1x1 = torch.nn.Conv1d(in_channels=in_channels+n_out_conv2,
                              out_channels=out_channels,
                              kernel_size=1,
                              bias=bias)
            if layerNorm is torch.nn.BatchNorm1d:
                self.norm1x1 = layerNorm(num_features=out_channels)
            elif isinstance(layerNorm, torch.nn.Identity):
                self.norm1x1 = layerNorm
            self.dropout1x1 = torch.nn.Dropout(dropout)

        if activation == "relu":
            self.activation =  torch.nn.functional.relu
        elif activation == "gelu":
            self.activation =  torch.nn.functional.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def forward(self, x, x_mask=None, x_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Parameters
        ----------
        x: tensor
            The input sequence to the encoder layer.
        x_mask: tensor
            Mask for the input sequence (optional).
        x_key_padding_mask: tensor
            Mask for the x keys per batch (optional).
        """
        z = self.dropout1(self.activation(self.norm1(self.conv1(x))))
        z = self.dropout2(self.activation(self.norm2(self.conv2(z))))
        if self.additive:
            x = x + z
        else:
            x = self.dropout1x1(self.activation(self.norm1x1(self.conv1x1(torch.cat((x, z), axis=1)))))

        return x


class ResNet(torch.nn.Module):

    def __init__(self, kernel_size, n_filters_ks4, filters_ks1_init=None, filters_ks1_inter=None,
                 filters_ks1_final=None, n_channels_in=1, n_channels_out=1,
                 padding_mode='zeros', additive=None, direct_shortcut=False,
                 layerNorm=torch.nn.BatchNorm1d, dropout=0.0):
        super(ResNet, self).__init__()
        self.kernel_size = kernel_size
        self.n_filters_ks1 = [filters_ks1_inter for i in range(len(n_filters_ks4) - 1)]
        self.n_filters_ks1 = self.n_filters_ks1 + [filters_ks1_final]
        if filters_ks1_init is None:
            self.n_filters_ks1 = [[]] + self.n_filters_ks1
        else:
            self.n_filters_ks1 = [filters_ks1_init] + self.n_filters_ks1
        assert len(self.n_filters_ks1) == len(n_filters_ks4) + 1

        self.direct_shortcut = direct_shortcut

        n_in = n_channels_in + 1
        self.layers4x4 = []
        self.layers_ks1 = [[] for i in range(len(self.n_filters_ks1))]
        for i in range(len(self.n_filters_ks1)):
            for j in range(len(self.n_filters_ks1[i])):
                n_out = self.n_filters_ks1[i][j]
                block = ResNetBlock(in_channels=n_in,
                                    kernel_size=self.kernel_size,
                                    hidden_channels=None,
                                    out_channels=n_out,
                                    bias=True,
                                    layerNorm=layerNorm,
                                    padding_mode='circular',
                                    dropout=dropout,
                                    activation="gelu",
                                    additive=additive if i & j != 0 else False)
                self.layers_ks1[i].append(block)
                n_in = n_out

            if i >= len(n_filters_ks4):
                break

            n_out = n_filters_ks4[i]
            layer = setup_conv1d(in_channels=n_in,
                               out_channels=n_out,
                               kernel_size=kernel_size,
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
        self.nonlinearity = torch.nn.GELU()

    def forward(self, xb, obs):
        x = torch.concat([xb, obs], dim=1)
        if self.direct_shortcut:
            out = xb

        for layer in self.layers_ks1[0]:
            x = self.nonlinearity(layer(x))
        for i, layer4x4 in enumerate(self.layers4x4):
            x = self.nonlinearity(layer4x4(x))
            for layer in self.layers_ks1[i + 1]:
                x = self.nonlinearity(layer(x))

        if self.direct_shortcut:
            return self.final(x) + out
        else:
            return self.final(x)