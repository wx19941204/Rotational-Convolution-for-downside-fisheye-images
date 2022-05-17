import math

import torch
import torch.nn as nn
import torchvision.ops

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 groups=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size[0],
                         kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.stride = stride

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * groups * kernel_size[0] *
                                     kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(in_channels,
                                   1 * groups * kernel_size[0] *
                                   kernel_size[1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=self.padding,
                                   bias=True)

        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        stride = self.stride

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          stride=stride,
                                          padding=self.padding,
                                          mask=mask)

        return x


# center constraint DeformableConv
class CC_DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 groups=1,
                 bias=False):

        super(CC_DeformableConv2d, self).__init__()
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size[0],
                         kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.stride = stride

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * groups * kernel_size[0] *
                                     kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(in_channels,
                                   1 * groups * kernel_size[0] *
                                   kernel_size[1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=self.padding,
                                   bias=True)

        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

        # nn.init.kaiming_normal_(self.offset_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = self.mask_conv(x)

        # fix center of deformable convolution
        offset[:, 8:10, ...] = 0
        mask[:, 4, ...] = 0
        mask = mask.sigmoid()

        # print("offset", offset[0, :, 0, 0])
        # print("mask", mask[0, :, 0, 0])

        stride = self.stride

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          stride=stride,
                                          padding=self.padding,
                                          mask=mask)

        return x