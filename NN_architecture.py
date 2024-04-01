import torch
import torch.nn as nn
import torch.nn.functional as F


class WangsNet(nn.Module):
    def __init__(self, pic_width, n_masks):
        super(WangsNet, self).__init__()
        m = n_masks
        self.pic_width = pic_width  # must be even number

        # first
        hidden_1, hidden_2 = self.pic_width**2, (2*self.pic_width)**2
        self.int_fc1 = nn.Linear(m, hidden_1)
        self.int_fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout = nn.Dropout(0.9)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, int(0.5*self.pic_width), 3, padding='same'),
            nn.BatchNorm2d(int(0.5*self.pic_width)))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(int(0.5*self.pic_width), self.pic_width, 3, padding='same'),
            nn.BatchNorm2d(self.pic_width))

        # fork
        self.conv_res = nn.Sequential(
            nn.Conv2d(self.pic_width, self.pic_width, 3, padding='same'),
            nn.BatchNorm2d(self.pic_width))

        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(4)
        self.maxpool8 = nn.MaxPool2d(8)
        self.upsample = nn.Upsample(scale_factor=2)

        # final
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(4*self.pic_width, 2*self.pic_width, 3, padding='same'),
            nn.BatchNorm2d(2*self.pic_width))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(2*self.pic_width, 2*self.pic_width, 3, padding='same'),
            nn.BatchNorm2d(2*self.pic_width))

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(2*self.pic_width, self.pic_width, 3, padding='same'),
            nn.BatchNorm2d(self.pic_width))

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.pic_width, 1, 3, padding='same'),
            nn.BatchNorm2d(1))

    def forward(self, x):
        x = self.first_block(x)
        x = self.fork_block(x)
        x = self.final_block(x)
        return x

    def first_block(self, x):
        x = F.relu(self.int_fc1(x))
        x = self.dropout(x)

        x = F.relu(self.int_fc2(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block1(x.view(-1, 1, 2*self.pic_width, 2*self.pic_width)))
        x = self.dropout(x)

        x = F.relu(self.conv_block2(x))
        x = self.dropout(x)

        return x

    def fork_block(self, x):
        # path 1
        x1 = self.res_block(x)

        # path 2
        x2 = self.maxpool2(x)
        x2 = self.res_block(x2)
        x2 = self.upsample(x2)

        # path 3
        x3 = self.maxpool4(x)
        x3 = self.res_block(x3)
        x3 = self.upsample(x3)
        x3 = self.upsample(x3)

        # path 4
        x4 = self.maxpool8(x)
        x4 = self.res_block(x4)
        x4 = self.upsample(x4)
        x4 = self.upsample(x4)
        x4 = self.upsample(x4)

        concat_x = torch.cat((x1, x2, x3, x4), 1)
        return concat_x

    def res_block(self, x):
        """ 4 blue res block, fit to all paths"""
        for _ in range(4):
            y = F.relu(self.conv_res(x))
            f_x = F.relu(self.conv_res(y))
            x = F.relu(x + f_x)
        return x

    def final_block(self, x):

        x = self.maxpool2(x)

        x = F.relu(self.conv_block3(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block4(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block5(x))
        x = self.dropout(x)

        x = F.relu(self.last_layer(x))

        return x




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