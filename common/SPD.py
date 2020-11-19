import torch
from torch import nn
from torch import FloatTensor
from torch.autograd import Variable
from torch import optim
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F
<<<<<<< HEAD
import scipy.stats as st
=======

>>>>>>> aceff018e185f10620599e45f313ed4f83412290
from tqdm import tqdm
from numpy.linalg import matrix_rank
from sklearn.datasets import make_spd_matrix
import os
# from astropy.convolution import discretize_model
from random import shuffle
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import transforms


def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma 
    sigma_y = sigma.float() / gamma 
    nstds = 3 
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax + 1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax + 1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float() 
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))
    gb = torch.exp(
        -.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb



class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=True)
        self.Lambda = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.psi = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        kernel = kernel.float()  
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        out = out + out.permute(0, 1, 3, 2) 

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


        return out

class GaborConv2d_improve(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False

        self.conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size
        self.delta = 1e-3
        self.freq = nn.Parameter(
            (math.pi / 2) * math.sqrt(2) ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True,
        )
        self.theta = nn.Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
        )
        self.sigma = nn.Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = nn.Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.x0 = nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ]
        )
        self.y = nn.Parameter(self.y)
        self.x = nn.Parameter(self.x)

        self.weight = nn.Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        self.calculate_weights()
        self.is_calculated = False
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                )
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.conv_layer.weight.data[i, j] = g

def th_atanh(x, EPS):
    values = torch.min(x, torch.Tensor([1.0 - EPS]))
    return 0.5 * (torch.log(1 + values + EPS) - torch.log(1 - values + EPS))


def th_norm(x, dim=1):
    return torch.norm(x, 2, dim, keepdim=True)


def th_dot(x, y, keepdim=True):
    return torch.sum(x * y, dim=1, keepdim=keepdim)


def clip_by_norm(x, clip_norm):
    return torch.renorm(x, 2, 0, clip_norm)



class Linear2D(nn.Module):

    def __init__(self, in_features=7, out_features=7, n_classes=1, k_size=3, dimm=14, EPS=1e-5, PROJ_EPS=1e-5):
        super(Linear2D, self).__init__()

  
        self.W = nn.Parameter(torch.rand((out_features, in_features)).normal_())  
        self.n_classes = n_classes
        self.dim1 = dimm
        self.dim2 = dimm
        self.EPS = EPS
        self.PROJ_EPS = PROJ_EPS
        self.tanh = nn.Tanh()
        self.ChannelAttention = nn.Sigmoid()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
    def normalize(self, x):
        return clip_by_norm(x, (1. - self.PROJ_EPS))

    def mob_add(self, u, v):
        v = v + self.EPS
        th_dot_u_v = 2. * th_dot(u, v)
        th_norm_u_sq = th_dot(u, u)
        th_norm_v_sq = th_dot(v, v)
        denominator = 1. + th_dot_u_v + th_norm_v_sq * th_norm_u_sq
        result = (1. + th_dot_u_v + th_norm_v_sq) / (denominator + self.EPS) * u + \
                 (1. - th_norm_u_sq) / (denominator + self.EPS) * v
        return self.normalize(result)

    def log_map_zero(self, y):
        diff = y + self.EPS
        norm_diff = th_norm(diff)
        return 1. / th_atanh(norm_diff, self.EPS) / norm_diff * diff

    def exp_map_zero(self, v):
        v = v + self.EPS
        norm_v = th_norm(v)  
        result = self.tanh(norm_v) / (norm_v) * v
        return self.normalize(result)

    def lorenz_factor(self, x, *, c=1.0, dim=-1, keepdim=False):
        return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))

    def p2k(self, x, c):
        denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
        return 2 * x / denom

    def k2p(self, x, c):
        denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
        return x / denom

    def poincare_mean(self, x, dim=-1, c=1.0):
        x = self.p2k(x, c)
        lamb = self.lorenz_factor(x, c=c, keepdim=True)
        mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(
            lamb, dim=dim, keepdim=True
        )
        mean = self.k2p(mean, c)
        return mean.view(self.dim0, -1, 1, 1)

    def forward(self, X):
        X_1 = X
        self.dim0 = X.size(0)  
        self.dim1 = X.size(1)  
        self.dim2 = X.size(-1)  
        X_1 = self.avg_pool(X)  
        X_1 = self.exp_map_zero(X_1)
        X_1 = self.poincare_mean(X_1)
        X_1 = self.log_map_zero(X_1)
        X_1 = self.conv1d(X_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        X_1 = self.ChannelAttention(X_1)
        X_1 = X_1.expand_as(X)
        X_1 = X_1.mul(X)
        X_1 = F.normalize(X_1)
        X = F.normalize(X)
        X = torch.cat((X, X_1), dim=1)
        return X

    def _matmul(self, X, Y, kind='right'):
        results = []
        for i in range(X.size(0)):
            if kind == 'right':
                result = torch.mm(X[i], Y)
            elif kind == 'left':
                result = torch.mm(Y, X[i])
            results.append(result.unsqueeze(0))
        return torch.cat(results, 0)

    def check_rank(self):
        return min(self.W.data.size()) == matrix_rank(self.W.data.numpy())
