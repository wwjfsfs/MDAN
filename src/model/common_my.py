import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size,bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class My_tail(nn.Module):
    def __init__(
        self, n_feats):

        super(My_tail, self).__init__()
        self.conv3 = nn.Conv2d(n_feats,n_feats,3,padding=3//2,bias=True)
        self.conv5 = nn.Conv2d(n_feats,n_feats,5,padding=5//2,bias=True)
        self.conv7 = nn.Conv2d(n_feats, n_feats, 7, padding=7 // 2, bias=True)
    def forward(self, x):
        res3 = self.conv3(x)
        res5 = self.conv5(x)
        res7 = self.conv7(x)
        res = res3+res5+res7
        return res

class upscale(nn.Module):
    def __init__(self, n_feats):
        super(upscale, self).__init__()
        up_2=[] #可以使用双线性插值完成
        up_2.append(nn.Conv2d(n_feats, 4 * n_feats,3,padding=1,bias=True))
        up_2.append(nn.PixelShuffle(2))
        self.up2 = nn.Sequential(*up_2)
    def forward(self,x):
        return self.up2(x)
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // 16, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 16, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res






# FAN: Frequency Aggregation Network for Real Image Super-resolution


class three_branch_v2(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(three_branch_v2, self).__init__()
        self.depose = depose(n_feats,kernel_size)


        self.h1 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h2 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h3 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.m1 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.m2 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.l1 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.mconv = conv(n_feats*2,n_feats,kernel_size=1)
        self.hconv = conv(n_feats*2,n_feats,kernel_size=1)
        self.conv1 = conv(n_feats*3,n_feats,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.esa = ESA(n_feats*3)#可以去掉加在最后cat部分
    def forward(self,x):
        xl,xm,xh = self.depose(x)
        _,_,h,w = xh.size()
        _,_,hm,wm = xm.size()
        xl = self.l1(xl)
        xl = torch.nn.functional.interpolate(xl, size=(hm,wm), scale_factor=None, mode='bilinear', align_corners=False)
        xm1 = self.m1(xm)
        #tmp = torch.cat([xm1,xl],dim=1)
        #tmp = self.mconv(tmp)
        xm2 = self.m2(self.mconv(torch.cat([xm1,xl],dim=1)))
        xm2 = torch.nn.functional.interpolate(xm2, size=(h, w), scale_factor=None, mode='bilinear', align_corners=False)
        xh = self.h1(xh)
        xh = self.h2(xh)
        #htmp = torch.cat([xh,xm2],dim=1)
        #htmp = self.hconv(htmp)

        xh = self.h3(self.hconv(torch.cat([xh,xm2],dim=1)))


        xl = torch.nn.functional.interpolate(xl, size=(h,w), scale_factor=None, mode='bilinear', align_corners=False)
        out = torch.cat([xh,xm2,xl],dim=1)
        out = self.esa(out)
        out = self.conv1(out)
        return out+self.gamma*x

class three_branch_v3(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(three_branch_v3, self).__init__()
        self.depose = depose(n_feats,kernel_size)


        self.h1 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h2 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h3 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.m1 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.m2 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.l1 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.mconv = conv(n_feats*2,n_feats,kernel_size=1)
        self.hconv = conv(n_feats*2,n_feats,kernel_size=1)
        self.conv1 = conv(n_feats*3,n_feats,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.esa = ESA(n_feats*3)#换成BAM注意力试一下
    def forward(self,x):
        xl,xm,xh = self.depose(x)
        _,_,h,w = xh.size()
        _,_,hm,wm = xm.size()
        xl = self.l1(xl)
        xl = torch.nn.functional.interpolate(xl,  scale_factor=2, mode='bicubic', align_corners=False)
        xm1 = self.m1(xm)
        #tmp = torch.cat([xm1,xl],dim=1)
        #tmp = self.mconv(tmp)
        xm2 = self.m2(self.mconv(torch.cat([xm1,xl],dim=1)))
        xm2 = torch.nn.functional.interpolate(xm2,  scale_factor=2, mode='bicubic', align_corners=False)
        xh = self.h1(xh)
        xh = self.h2(xh)
        #htmp = torch.cat([xh,xm2],dim=1)
        #htmp = self.hconv(htmp)

        xh = self.h3(self.hconv(torch.cat([xh,xm2],dim=1)))


        xl = torch.nn.functional.interpolate(xl,  scale_factor=2, mode='bicubic', align_corners=False)
        out = torch.cat([xh,xm2,xl],dim=1)
        out = self.esa(out)
        out = self.conv1(out)
        return out+self.gamma*x

class PA(nn.Module):
    def __init__(self,nf):
        super(PA,self).init()
        self.conv = nn.Conv2d(nf,nf,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x,y)
        return out



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

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ChannelAttention_maxpool(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_maxpool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.PReLU(in_planes // ratio)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out+max_out
        return self.sigmoid(out)

class SpatialAttention_averagepool(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_averagepool, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BlancedAttention_CAM_SAM_ADD(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(BlancedAttention_CAM_SAM_ADD, self).__init__()

        self.ca = ChannelAttention_maxpool(in_planes, reduction)
        self.sa = SpatialAttention_averagepool()
        # self.conv=nn.Conv2d(in_planes*2,in_planes,1)
        # self.norm=nn.BatchNorm2d(in_planes)

    def forward(self, x):
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out=ca_ch.mul(sa_ch)*x
        # out_fused = self.conv(torch.cat([ca_ch, sa_ch], dim=1))
        return out

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class INB(nn.Module):
    def __init__(
        self, conv, n_feats):

        super(INB, self).__init__()
        self.conv1 = conv(n_feats,n_feats,kernel_size=3)
        #self.convr = conv(n_feats, n_feats, kernel_size=3)
        self.convl = conv(n_feats,n_feats//2,kernel_size=1)
        self.ca = CALayer(n_feats)
        self.pa = PA(n_feats)
        self.conv2 = conv(n_feats+n_feats//2,n_feats,kernel_size=1)
        self.act=activation('lrelu', neg_slope=0.05)
        self.IN = norm('instance',n_feats//2)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.act(x+x1)
        x2 = self.convl(x)
        x2 = self.act(self.IN(x2))
        res = torch.cat([x1,x2],dim=1)
        res = self.pa(self.conv2(res))
        res = res+self.ca(x)

        return res
class LAM_Module(nn.Module):
    """ Layer attention module"""

    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1).contiguous()
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        return out
class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""

    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                antialias.Downsample(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(antialias.Downsample(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

import numpy as np
class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,
                                                    bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x
class depose(nn.Module):
    def __init__(self, n_feats,kernel_size):
        super(depose, self).__init__()
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                              nn.Conv2d(n_feats, n_feats, kernel_size=7, padding=0),
                              nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, stride=2, padding=1),
                              nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, stride=2, padding=1),
                              nn.ReLU(True))


    def forward(self,x):

        x_down1 = self.down1(x)  # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1)  # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2)
        return x_down3 , x_down2, x_down1  #n*4,n*2,n


class three_branch_v4(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(three_branch_v4, self).__init__()
        self.depose = depose(n_feats,kernel_size)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 4, n_feats * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up4 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(n_feats, n_feats, kernel_size=7, padding=0),
                                 nn.ReLU(True))
        self.h1 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h2 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h3 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.m1 = RCAB(conv, n_feats*2, kernel_size, act=act, res_scale=1)
        self.m2 = RCAB(conv, n_feats*2, kernel_size, act=act, res_scale=1)
        self.l1 = RCAB(conv, n_feats*4, kernel_size, act=act, res_scale=1)

        self.esa = ESA(n_feats*3)#换成BAM注意力试一下
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv1 = conv(n_feats * 3, n_feats, kernel_size=1)
    def forward(self,x):
        xl,xm,xh = self.depose(x)

        xl = self.l1(xl) #b,c*4,h/4,w/4
        xlm = self.up1(xl) # b,c*2,h/2,w/2
        xm1 = self.m1(xm)   #b,c*2,h/2,w/2

        xm2 = self.m2(xm1)#b,c*2,h/2,w/2
        xmh = self.up2(xm2)#b,c,h,w
        xh = self.h1(xh)
        xh = self.h2(xh)


        xh = self.h3(xh) #b,c,h,w
        xh = self.up4(xh)
        xl = self.up3(xlm)

        out = torch.cat([xh,xmh,xl],dim=1)
        out = self.esa(out)
        out = self.conv1(out)
        return out+self.gamma*x



class pan_three_branch_v3(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(pan_three_branch_v3, self).__init__()
        self.depose = depose_v2(n_feats)
        unc = 24
        self.upconv1 = nn.Conv2d(n_feats,unc,3,1,1,bias=True)
        self.pa1 = PA(unc)
        self.HRconv1 = nn.Conv2d(unc,n_feats,3,1,1,bias=True)
        self.upconv2 = nn.Conv2d(n_feats,unc,3,1,1,bias=True)
        self.pa2 = PA(unc)
        self.HRconv2 = nn.Conv2d(unc,n_feats,3,1,1,bias=True)
        self.h1 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h2 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.h3 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.m1 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.m2 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.l1 = RCAB(conv, n_feats, kernel_size, act=act, res_scale=1)
        self.mconv = conv(n_feats*2,n_feats,kernel_size=1)
        self.hconv = conv(n_feats*2,n_feats,kernel_size=1)
        self.conv1 = conv(n_feats*3,n_feats,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.esa = ESA(n_feats*3)#换成BAM注意力试一下
        self.lrelu = nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self,x):
        xl,xm,xh = self.depose(x)
        _,_,h,w = xh.size()
        _,_,hm,wm = xm.size()
        xl = self.l1(xl)
        xl = torch.nn.functional.interpolate(xl, size=(hm,wm), scale_factor=None, mode='bilinear', align_corners=False)
        xm1 = self.m1(xm)

        xm2 = self.m2(self.mconv(torch.cat([xm1,xl],dim=1)))
        xm2 = torch.nn.functional.interpolate(xm2, size=(h, w), scale_factor=None, mode='bilinear', align_corners=False)
        xh = self.h1(xh)
        xh = self.h2(xh)


        xh = self.h3(self.hconv(torch.cat([xh,xm2],dim=1)))


        xl = torch.nn.functional.interpolate(xl, size=(h,w), scale_factor=None, mode='bilinear', align_corners=False)
        xl_out = self.upconv1(xl)
        xl_out = self.lrelu(self.pa1(xl_out))
        xl_out = self.lrelu(self.HRconv1(xl_out))
        xm_out = self.upconv2(xm2)
        xm_out = self.lrelu(self.pa2(xm_out))
        xm_out = self.lrelu(self.HRconv2(xm_out))
        out = torch.cat([xh,xm_out,xl_out],dim=1)
        out = self.esa(out)
        out = self.conv1(out)
        return out+self.gamma*x

class depose_v2(nn.Module):
    def __init__(self, n_feats):
        super(depose_v2, self).__init__()
        self.conv_l = nn.Conv2d(n_feats , n_feats , kernel_size=3, padding=1, stride=2,
                                bias=True)  # or two conv stride=2
        self.conv_m = nn.Conv2d(n_feats, n_feats , kernel_size=5, padding=2, stride=2, bias=True)
        self.conv_h = nn.Conv2d(n_feats, n_feats, kernel_size=7, padding=3, stride=1, bias=True)

    def forward(self, x):
        xh = self.conv_h(x)

        xm = self.conv_m(xh)

        xl = self.conv_l(xm)

        return xl, xm, xh


class high_branch(nn.Module):
    def __init__(
            self, conv, n_feats):
        super(high_branch, self).__init__()

        h = [
            INB(
                conv, n_feats
            ) for _ in range(10)
        ]

        self.h = nn.Sequential(*h)

        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats * 10, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
    def forward(self, x):
        res = x
        res1 = []
        # pdb.set_trace()
        for name, midlayer in self.h._modules.items():
            res = midlayer(res)
            # print(name)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1), res1], 1)
        # res = self.body(x)
        out1 = res
        # res3 = res.unsqueeze(1)
        # res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        res = res + x
        return res


class mid_branch(nn.Module):
    def __init__(
            self, conv, n_feats):
        super(mid_branch, self).__init__()

        m = [
            INB(
                conv, n_feats
            ) for _ in range(8)
        ]


        self.m = nn.Sequential(*m)

        # 可以去掉加在最后cat部分

        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats * 8, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)

    def forward(self, x):
        res = x
        res1 = []
        # pdb.set_trace()
        for name, midlayer in self.m._modules.items():
            res = midlayer(res)
            # print(name)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1), res1], 1)
        # res = self.body(x)
        out1 = res
        # res3 = res.unsqueeze(1)
        # res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        res = res + x
        return res


class low_branch(nn.Module):
    def __init__(
            self, conv, n_feats):
        super(low_branch, self).__init__()

        l = [
            INB(
                conv, n_feats
            ) for _ in range(6)
        ]

        self.l = nn.Sequential(*l)
        # 可以去掉加在最后cat部分

        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats * 6, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)

    def forward(self, x):
        res = x
        res1 = []
        # pdb.set_trace()
        for name, midlayer in self.l._modules.items():
            res = midlayer(res)
            # print(name)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1), res1], 1)
        # res = self.body(x)
        out1 = res
        # res3 = res.unsqueeze(1)
        # res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        res = res + x
        return res

class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        # for pytorch 0.3.1
        # nn.init.constant(self.W.weight, 0)
        # nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


## define trunk branch
class TrunkBranch(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        tx = self.body(x)

        return tx


class NLMaskBranchDownUp(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLMaskBranchDownUp, self).__init__()

        MB_RB1 = []
        MB_RB1.append(NonLocalBlock2D(n_feat, n_feat // 2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        # MB_Up = []
        # MB_Up.append(nn.ConvTranspose2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        #self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        _,_,h,w = x_RB1.size()
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        #x_Up = self.MB_Up(x_RB2)
        x_Up = torch.nn.functional.interpolate(x_RB2, size=(h,w), scale_factor=None, mode='bilinear', align_corners=False)
        # print(x_Up.shape)
        # print(x_RB1.shape)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


class NLResAttModuleDownUpPlus(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx

