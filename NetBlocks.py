from torch import nn
from torch.nn import functional as F
import torch


def conv_block(inputs, n):
    x = F.conv2d(inputs)
    x = nn.BatchNorm1d(x)
    x = F.relu(x)
    return x


def res_block(inputs, n):
    x = conv_block(inputs, n)
    x = conv_block(x, n)
    x = torch.add([inputs, x])
    return x


def initial_block(inputs):
    x = nn.Linear(28*28, hidden_1)
    x = F.relu(x)
    x = nn.Dropout(0.5)(x)
    x = nn.Linear(4 * s1 * s2, x)
    x = F.relu(x)
    x = nn.Dropout(0.5)(x)
    x = x.view([64, 64, 1])
    x = conv_block(x, 16)
    x = nn.Dropout(0.5)(x)
    x = conv_block(x, 32)
    init_block_out = nn.Dropout(0.5)(x)
    return init_block_out


def fork_1(block_1_out, s=32):
    x = res_block(block_1_out, s)
    x = res_block(x, s)
    x = res_block(x, s)
    path_1_out = res_block(x, s)
    return path_1_out


def fork_2(block_1_out, s=32):
    x = nn.MaxPool2d(2)(block_1_out)
    x = res_block(x, s)
    x = res_block(x, s)
    x = res_block(x, s)
    x = res_block(x, s)
    path_2_out = nn.UpsamplingNearest2d([2, 2])(x)
    return path_2_out


def fork_3(block_1_out, s=32):
    x = nn.MaxPool2d(2)(block_1_out)
    x = res_block(x, s)
    x = res_block(x, s)
    x = res_block(x, s)
    x = res_block(x, s)
    x = nn.UpsamplingNearest2d([2, 2])(x)
    path_3_out = nn.UpsamplingNearest2d([2, 2])(x)
    return path_3_out


def fork_4(block_1_out, s=32):
    x = nn.MaxPool2d(2)(block_1_out)
    x = res_block(x, s)
    x = res_block(x, s)
    x = res_block(x, s)
    x = res_block(x, s)
    x = nn.UpsamplingNearest2d([2, 2])(x)
    x = nn.UpsamplingNearest2d([2, 2])(x)
    path_4_out = nn.UpsamplingNearest2d([2, 2])(x)
    return path_4_out


def fork_block(init_block_out):
    path_1_out = fork_1(init_block_out)
    path_2_out = fork_2(init_block_out)
    path_3_out = fork_3(init_block_out)
    path_4_out = fork_4(init_block_out)
    fork_block_out = [path_1_out, path_2_out, path_3_out, path_4_out]
    return fork_block_out


def final_block(fork_block_out):
    x = torch.cat(fork_block_out)
    x = conv_block(x, 64)
    x = nn.MaxPool2d(2)(x)
    x = conv_block(x, 64)
    x = conv_block(x, 32)
    final_block_out = conv_block(x, 1)
    return final_block_out












