import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import torchvision
import math
from torch.nn.modules.utils import _pair
from groupy.gconv.make_gconv_indices import *
from RotConv.group_conv_mask import group_conv_mask, group_conv_8mask, group_conv_2mask
from torchvision.transforms.functional import rotate as TorchRotate
import cv2

# indices for rotating filter
make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices}


img_size = 512
mask = group_conv_mask(input_size=img_size)   # mask for RC4
mask8 = group_conv_8mask(input_size=img_size)  # mask for RC8
mask2 = group_conv_2mask(input_size=img_size)  # mask for RC2


def trans_filter(w, inds):
    # same with trans_filter_2.  faster!
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()

# same with trans_filter
def trans_filter_2(w):
    w_90 = TorchRotate(w, angle=90)
    w_180 = TorchRotate(w, angle=180)
    w_270 = TorchRotate(w, angle=270)
    tw = torch.cat([w, w_90, w_180, w_270], dim=0)
    return tw


class RotateCCDeformConv2D(nn.Module):
    # Rotational Convolution based on 4 states with deformable convolution v2
    # Here the center of deformable convolution is fixed.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, input_stabilizer_size=1, output_stabilizer_size=4):
        super(RotateCCDeformConv2D, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size
        self.mask = Parameter(torch.Tensor(mask), requires_grad=False)
        self.act = nn.ReLU(inplace=True)

        # weight of Rotational Convolution
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

        # deformableconv params
        deform_inchannls = in_channels * input_stabilizer_size
        self.offset_conv = nn.Conv2d(deform_inchannls,
                                     2 * kernel_size[0] *
                                     kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(deform_inchannls,
                                   1 * kernel_size[0] *
                                   kernel_size[1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=self.padding,
                                   bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        # rotate the origin weight to 4 states and concat them.
        tw = trans_filter(self.weight, self.inds)
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)

        input_shape = input.size()
        h, w = input_shape[-2], input_shape[-1]
        assert h == w, "For fisheye images, h and w must be equal."

        # downsample the mask to the size of output
        (h, w) = (h // 2, w // 2) if self.stride[0] == 2 else (h, w)
        mask = nn.AdaptiveAvgPool2d((h, w))(self.mask)

        # copy input into 4 copies
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])

        # deformable conv
        deform_offset = self.offset_conv(input)
        deform_mask = self.mask_conv(input)

        # fix the center of deformable convolution
        deform_offset[:, 8:10, ...] = 0
        # print(mask.shape)
        deform_mask[:, 4, ...] = 0
        deform_mask = deform_mask.sigmoid()

        y = torchvision.ops.deform_conv2d(input,
                                          offset=deform_offset,
                                          weight=tw,
                                          bias=None,
                                          stride=self.stride,
                                          padding=self.padding,
                                          mask=deform_mask)

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        y = self.act(y)
        y = y * mask
        y = y.sum(dim=2)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias

        return y


class Rotate_8_CCDeformConv2D_V1(nn.Module):
    # # Rotational Convolution based on 8 states with deformable convolution v2
    #     # Here the center of deformable convolution is fixed.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, input_stabilizer_size=1, output_stabilizer_size=8):
        super(Rotate_8_CCDeformConv2D_V1, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size
        self.mask = Parameter(torch.Tensor(mask8), requires_grad=False)
        self.act = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=11)
        self.downsample = nn.AdaptiveAvgPool2d((3, 3))

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

        # deformableconv params
        deform_inchannls = in_channels * input_stabilizer_size
        self.offset_conv = nn.Conv2d(deform_inchannls,
                                     2 * kernel_size[0] *
                                     kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(deform_inchannls,
                                   1 * kernel_size[0] *
                                   kernel_size[1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=self.padding,
                                   bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size // 2)](self.ksize)


    def forward(self, input):

        tw = trans_filter(self.weight, self.inds)
        weight45 = self.weight[:, :, 0, ...]
        weight45 = TorchRotate(weight45, angle=45)

        weight45 = torch.unsqueeze(weight45, dim=2)
        tw2 = trans_filter(weight45, self.inds)

        tw = torch.cat((tw, tw2), dim=1)
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)


        input_shape = input.size()
        h, w = input_shape[-2], input_shape[-1]
        assert h == w, "h and w must be equal."

        (h, w) = (h // 2, w // 2) if self.stride[0] == 2 else (h, w)
        mask = nn.AdaptiveAvgPool2d((h, w))(self.mask)

        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])

        # deformable conv
        deform_offset = self.offset_conv(input)
        deform_mask = self.mask_conv(input)

        # fix the center of deformable convolution
        deform_offset[:, 8:10, ...] = 0
        deform_mask[:, 4, ...] = 0
        deform_mask = deform_mask.sigmoid()

        y = torchvision.ops.deform_conv2d(input,
                                          offset=deform_offset,
                                          weight=tw,
                                          bias=None,
                                          stride=self.stride,
                                          padding=self.padding,
                                          mask=deform_mask)

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        y = self.act(y)
        y = y * mask
        y = y.sum(dim=2)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias

        return y


class Rotate_2_CCDeformConv2D(nn.Module):
    # # RC2:Rotational Convolution based on 2 states with deformable convolution v2
    #     # Here the center of deformable convolution is fixed.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, input_stabilizer_size=1, output_stabilizer_size=2):
        super(Rotate_2_CCDeformConv2D, self).__init__()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size
        self.mask = Parameter(torch.Tensor(mask2), requires_grad=False)
        self.act = nn.ReLU(inplace=True)

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

        # deformableconv params
        deform_inchannls = in_channels * input_stabilizer_size
        self.offset_conv = nn.Conv2d(deform_inchannls,
                                     2 * kernel_size[0] *
                                     kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.mask_conv = nn.Conv2d(deform_inchannls,
                                   1 * kernel_size[0] *
                                   kernel_size[1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=self.padding,
                                   bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(1, 4)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds)
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)

        tw = torch.cat((tw[:, 0, ...], tw[:, 2, ...]), dim=0)
        tw = tw.view(tw_shape)

        input_shape = input.size()
        h, w = input_shape[-2], input_shape[-1]
        assert h == w, "h and w must be equal."

        (h, w) = (h // 2, w // 2) if self.stride[0] == 2 else (h, w)
        mask = nn.AdaptiveAvgPool2d((h, w))(self.mask)

        input = input.view(input_shape[0], self.in_channels * self.input_stabilizer_size, input_shape[-2],
                           input_shape[-1])
        # deformable conv
        deform_offset = self.offset_conv(input)
        deform_mask = self.mask_conv(input)

        # fix the center of deformable convolution
        deform_offset[:, 8:10, ...] = 0
        deform_mask[:, 4, ...] = 0
        deform_mask = deform_mask.sigmoid()

        y = torchvision.ops.deform_conv2d(input,
                                          offset=deform_offset,
                                          weight=tw,
                                          bias=None,
                                          stride=self.stride,
                                          padding=self.padding,
                                          mask=deform_mask)
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        y = self.act(y)
        y = y * mask
        y = y.sum(dim=2)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias

        return y