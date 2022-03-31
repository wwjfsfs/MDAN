import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


def make_model(args):
    return MDAN(args)  # 空洞率改变


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.group_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 1, groups=self.groups))
        self.depth_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1, groups=in_channels))
        self.point_conv = wn(nn.Conv2d(self.in_channels, self.out_channels, 1, groups=1))

    def forward(self, x):
        x = self.group_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ConvBlockD(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3, ker_size=2):
        super(ConvBlockD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.group_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 1, groups=self.groups))
        self.depth_conv = wn(
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding=ker_size, dilation=ker_size, groups=in_channels))
        self.point_conv = wn(nn.Conv2d(self.in_channels, self.out_channels, 1, groups=1))

    def forward(self, x):
        x = self.group_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MIRB1(nn.Module):
    def __init__(self, args):
        super(MIRB1, self).__init__()
        self.c_out = args.n_feats // 2
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv3_1 = ConvBlock(args.n_feats, self.c_out)
        self.convd_1 = ConvBlock(args.n_feats, self.c_out)

        self.conv3_2 = ConvBlock(args.n_feats, self.c_out)
        self.convd_2 = ConvBlock(args.n_feats, self.c_out)

        self.conv3_3 = ConvBlock(args.n_feats, self.c_out)
        self.convd_3 = ConvBlock(args.n_feats, self.c_out)
        self.conv_last = wn(nn.Conv2d(args.n_feats, args.n_feats, 1))
        # self.calayer = CALayer(args.n_feats)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        res = x
        c1_1 = self.lrelu(self.conv3_1(res))
        c2_1 = self.lrelu(self.convd_1(res))

        c1_2 = self.lrelu(self.conv3_2(torch.cat([c1_1, c2_1], 1)))
        c2_2 = self.lrelu(self.convd_2(torch.cat([c1_1, c2_1], 1)))

        c1_4 = self.lrelu(self.conv3_3(torch.cat([c1_2, c2_2], 1)))
        c2_4 = self.lrelu(self.convd_3(torch.cat([c1_2, c2_2], 1)))

        out = self.conv_last(torch.cat([c1_4, c2_4], 1))
        out = out + x
        return out


class MIRB2(nn.Module):
    def __init__(self, args):
        super(MIRB2, self).__init__()
        self.c_out = args.n_feats // 2
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.conv3_1 = ConvBlock(args.n_feats, self.c_out)
        self.convd_1 = ConvBlockD(args.n_feats, self.c_out, ker_size=2)

        self.conv3_2 = ConvBlock(args.n_feats, self.c_out)
        self.convd_2 = ConvBlockD(args.n_feats, self.c_out, ker_size=2)

        self.conv3_3 = ConvBlock(args.n_feats, self.c_out)
        self.convd_3 = ConvBlockD(args.n_feats, self.c_out, ker_size=2)
        self.conv_last = wn(nn.Conv2d(args.n_feats, args.n_feats, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        res = x
        c1_1 = self.lrelu(self.conv3_1(res))
        c2_1 = self.lrelu(self.convd_1(res))

        c1_2 = self.lrelu(self.conv3_2(torch.cat([c1_1, c2_1], 1)))
        c2_2 = self.lrelu(self.convd_2(torch.cat([c1_1, c2_1], 1)))

        c1_4 = self.lrelu(self.conv3_3(torch.cat([c1_2, c2_2], 1)))
        c2_4 = self.lrelu(self.convd_3(torch.cat([c1_2, c2_2], 1)))

        out = self.conv_last(torch.cat([c1_4, c2_4], 1))
        # out = self.calayer(out)
        out = out + x
        return out


class MIRB3(nn.Module):
    def __init__(self, args):
        super(MIRB3, self).__init__()
        self.c_out = args.n_feats // 2
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.conv3_1 = ConvBlock(args.n_feats, self.c_out)
        self.convd_1 = ConvBlockD(args.n_feats, self.c_out, ker_size=3)

        self.conv3_2 = ConvBlock(args.n_feats, self.c_out)
        self.convd_2 = ConvBlockD(args.n_feats, self.c_out, ker_size=3)

        self.conv3_3 = ConvBlock(args.n_feats, self.c_out)
        self.convd_3 = ConvBlockD(args.n_feats, self.c_out, ker_size=3)
        self.conv_last = wn(nn.Conv2d(args.n_feats, args.n_feats, 1))
        # self.calayer = CALayer(args.n_feats)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        res = x
        c1_1 = self.lrelu(self.conv3_1(res))
        c2_1 = self.lrelu(self.convd_1(res))

        c1_2 = self.lrelu(self.conv3_2(torch.cat([c1_1, c2_1], 1)))
        c2_2 = self.lrelu(self.convd_2(torch.cat([c1_1, c2_1], 1)))

        c1_4 = self.lrelu(self.conv3_3(torch.cat([c1_2, c2_2], 1)))
        c2_4 = self.lrelu(self.convd_3(torch.cat([c1_2, c2_2], 1)))
        out = self.conv_last(torch.cat([c1_4, c2_4], 1))
        # out = self.calayer(out)
        out = out + x
        return out


class MMFB(nn.Module):
    def __init__(self, args):
        super(MMFB, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        # self.n_feats = args.n_feats
        self.bs1 = MIRB1(args)
        self.bs11 = MIRB1(args)
        self.bs2 = MIRB2(args)
        self.bs22 = MIRB2(args)
        self.bs3 = MIRB3(args)
        self.bs33 = MIRB3(args)

    def forward(self, x):
        res = x
        #res_csb = x.clone()
        res = self.bs1(res)
        res = self.bs11(res)
        res = self.bs2(res)
        res = self.bs22(res)
        res = self.bs3(res)
        res = self.bs33(res)
        out = res + x
        return out


class MDAB(nn.Module):
    def __init__(self, args):
        super(MDAB, self).__init__()
        self.n_feats = args.n_feats
        self.scale = args.scale[0]
        self.out_feats = self.scale * self.scale * args.n_colors
        self.im_feats = self.out_feats * 2  # 2:24,3:54,4:96

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.tail1 = wn(nn.Conv2d(args.n_feats, args.n_feats // 2, 1))
        self.tail2 = ConvBlock(args.n_feats, args.n_feats // 2)

        self.conv = wn(nn.Conv2d(args.n_feats, args.n_feats, 1))
        self.conv3 = ConvBlock(args.n_feats, args.n_feats)
        self.soft_c = nn.Softmax(dim=1)
        self.soft_hw = nn.Softmax(dim=2)
        self.conv_end = wn(nn.Conv2d(args.n_feats, args.n_feats, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.tail1(x)
        x2 = self.tail2(x)
        x_c = torch.cat([x1, x2], 1)
        x_r = self.conv(x_c)
        x_a = self.conv3(x_c)
        x_a1 = self.soft_c(x_a)
        x_a2 = self.soft_hw(x_a.view(b, c, -1))
        x_a2 = x_a2.reshape(b, c, h, w)

        out = x_r * x_a1 + x_r * x_a2
        out = self.conv_end(out)
        return out


class MDAN(nn.Module):
    def __init__(self, args):
        super(MDAN, self).__init__()
        # n_resgroups = args.n_resgroups
        self.n_feats = args.n_feats
        self.scale = args.scale[0]
        self.out_feats = self.scale * self.scale * args.n_colors
        self.im_feats = self.out_feats * 2  # 2:24,3:54,4:96

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.sub_mean = MeanShift(3)
        self.up_bic = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv_first = wn(nn.Conv2d(3, self.n_feats, 3, padding=1))

        self.BS1 = MMFB(args)
        self.BS2 = MMFB(args)
        self.BS3 = MMFB(args)
        self.upb1 = MDAB(args)
        self.upb2 = MDAB(args)
        self.upb3 = MDAB(args)

        self.scale1 = Scale(0.3)
        self.scale2 = Scale(0.3)
        self.scale3 = Scale(0.4)
        self.conv_add = wn(nn.Conv2d(args.n_feats * 3, args.n_feats, 1))
        #self.out = wn(nn.Conv2d(args.n_feats, self.scale * self.scale * 3, 3, padding=1)) # 2x
        self.out1 = wn(nn.Conv2d(args.n_feats, self.scale * self.scale * 3, 3, padding=1)) # 3x,4x
        self.pixelshuffle = nn.PixelShuffle(self.scale)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.add_mean = MeanShift(3, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x_id = x
        x_id = self.up_bic(x_id)
        x = self.lrelu(self.conv_first(x))
        res = x
        res = self.BS1(res)
        res1 = self.BS2(res)
        res2 = self.BS3(res1)

        out1 = self.scale1(self.upb1(res))
        out2 = self.scale2(self.upb2(res1))
        out3 = self.scale3(self.upb3(res2))

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv_add(out)

        out = out + x
        #out = self.out(out) # 2x
        out = self.out1(out) # 3x,4x
        out = self.pixelshuffle(out)
        out = out + x_id
        out = self.add_mean(out)
        return out


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import sys
    import numpy as np
    from thop import profile

    sys.path.append("../")
    from option import args

    args.scale = [4]
    # from option import args
    args.n_feats = 48
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MDAN(args).to(device)
    # 4倍 320/180  2倍  640/360  3倍  427/240
    input = torch.randn(1, 3, 40, 45).cuda()
    # input = torch.randn(16, 3, 48, 48).cuda()
    output = model(input)
    print(output.shape)


    def params_count(model):
        """
        Compute the number of parameters.
        Args:
            model (model): model to count the number of parameters.
        """
        return np.sum([p.numel() for p in model.parameters()]).item()


    print("params:", params_count(model))
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)
