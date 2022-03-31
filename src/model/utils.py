import torch
import torch.nn as nn
import torch.nn.functional as F


# 去雾论文中的求梯度方式
def gradient(y):
    gradient_h = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    return gradient_h, gradient_y


criterionCAE = nn.L1Loss()
criterionCAE.cude()


lOSS_H = criterionCAE()
LOSS_Y = criterionCAE()


# TSAN中的no-local
class CSB(nn.Module):
    def __init__(self, in_ch):
        super(CSB, self).__init__()
        self.conv1 = nn.Conv2d(in_ch*4, in_ch*4, 1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        if h % 2 == 1:
            i = 1
        else:
            i = 0
        if w % 2 == 1:
            j = 1
        else:
            j = 0
        f1 = x[:, :, 0:x.size(2) // 2+i, 0:x.size(3) // 2+j]
        f2 = x[:, :, x.size(2) // 2:, 0:x.size(3) // 2+j]
        f3 = x[:, :, 0:x.size(2) // 2+i, x.size(3) // 2:]
        f4 = x[:, :, x.size(2) // 2:, x.size(3) // 2:]
        f_all = torch.cat([f1, f2, f3, f4], 1)
        f_all = self.conv1(f_all)
        out = x
        out[:, :, 0:x.size(2) // 2, 0:x.size(3) // 2] = f_all[:, 0:f_all.size(1) * 1 // 4, 0:x.size(2) // 2, 0:x.size(3) // 2]
        out[:, :, x.size(2) // 2:, 0:x.size(3) // 2] = f_all[:, f_all.size(1) * 1 // 4:f_all.size(1) * 2 // 4, :, 0:x.size(3) // 2]
        out[:, :, 0:x.size(2) // 2, x.size(3) // 2:] = f_all[:, f_all.size(1) * 2 // 4:f_all.size(1) * 3 // 4, 0:x.size(2) // 2, :]
        out[:, :, x.size(2) // 2:, x.size(3) // 2:] = f_all[:, f_all.size(1) * 3 // 4:f_all.size(1) * 4 // 4, :, :]
        return out


class MYCSB(nn.Module):
    def __init__(self, in_ch):
        super(MYCSB, self).__init__()
        self.c_sub = in_ch // 2
        self.conv_up1 = nn.Conv2d((in_ch // 2) * 4, in_ch // 2, 1, stride=1, padding=0)
        self.conv_up2 = nn.Conv2d(in_ch // 2, (in_ch // 2) * 4, 1, stride=1, padding=0)
        self.conv_down1 = nn.Conv2d((in_ch // 2) * 4, in_ch // 2, 1, stride=1, padding=0)
        self.conv_down2 = nn.Conv2d(in_ch // 2, in_ch // 2, 3, stride=1, padding=1)
        self.conv_down3 = nn.Conv2d(in_ch // 2, (in_ch // 2) * 4, 1, stride=1, padding=0)

    def forward(self, x):
        x1 = x[:, 0:self.c_sub, :, :]
        x2 = x[:, self.c_sub:, :, :]
        b1, c1, h1, w1 = x1.size()
        b2, c2, h2, w2 = x2.size()
        if h1 % 2 == 1:
            i1 = 1
        else:
            i1 = 0
        if w1 % 2 == 1:
            j1 = 1
        else:
            j1 = 0
        if h2 % 2 == 1:
            i2 = 1
        else:
            i2 = 0
        if w2 % 2 == 1:
            j2 = 1
        else:
            j2 = 0
        f1_1 = x1[:, :, 0:x1.size(2) // 2 + i1, 0:x1.size(3) // 2 + j1]
        f2_1 = x1[:, :, x1.size(2) // 2:, 0:x1.size(3) // 2 + j1]
        f3_1 = x1[:, :, 0:x1.size(2) // 2 + i1, x1.size(3) // 2:]
        f4_1 = x1[:, :, x1.size(2) // 2:, x1.size(3) // 2:]
        f_sub1 = torch.cat([f1_1, f2_1, f3_1, f4_1], 1)
        f_sub1 = self.conv_up1(f_sub1)
        f_sub1 = self.conv_up2(f_sub1)
        out1 = x1
        out1[:, :, 0:x1.size(2) // 2, 0:x1.size(3) // 2] = f_sub1[:, 0:f_sub1.size(1) * 1 // 4, 0:x1.size(2) // 2,
                                                        0:x1.size(3) // 2]
        out1[:, :, x1.size(2) // 2:, 0:x1.size(3) // 2] = f_sub1[:, f_sub1.size(1) * 1 // 4:f_sub1.size(1) * 2 // 4, :,
                                                       0:x1.size(3) // 2]
        out1[:, :, 0:x1.size(2) // 2, x1.size(3) // 2:] = f_sub1[:, f_sub1.size(1) * 2 // 4:f_sub1.size(1) * 3 // 4,
                                                       0:x1.size(2) // 2, :]
        out1[:, :, x1.size(2) // 2:, x1.size(3) // 2:] = f_sub1[:, f_sub1.size(1) * 3 // 4:f_sub1.size(1) * 4 // 4, :, :]

        f1_2 = x2[:, :, 0:x2.size(2) // 2 + i2, 0:x2.size(3) // 2 + j2]
        f2_2 = x2[:, :, x2.size(2) // 2:, 0:x2.size(3) // 2 + j2]
        f3_2 = x2[:, :, 0:x2.size(2) // 2 + i2, x2.size(3) // 2:]
        f4_2 = x2[:, :, x2.size(2) // 2:, x2.size(3) // 2:]
        f_sub2 = torch.cat([f1_2, f2_2, f3_2, f4_2], 1)
        f_sub2 = self.conv_down1(f_sub2)
        f_sub2 = self.conv_down2(f_sub2)
        f_sub2 = self.conv_down3(f_sub2)
        out2 = x2
        out2[:, :, 0:x2.size(2) // 2, 0:x2.size(3) // 2] = f_sub2[:, 0:f_sub2.size(1) * 1 // 4, 0:x2.size(2) // 2,
                                                       0:x2.size(3) // 2]
        out2[:, :, x2.size(2) // 2:, 0:x2.size(3) // 2] = f_sub2[:, f_sub2.size(1) * 1 // 4:f_sub2.size(1) * 2 // 4, :,
                                                          0:x2.size(3) // 2]
        out2[:, :, 0:x2.size(2) // 2, x2.size(3) // 2:] = f_sub2[:, f_sub2.size(1) * 2 // 4:f_sub2.size(1) * 3 // 4,
                                                          0:x2.size(2) // 2, :]
        out2[:, :, x2.size(2) // 2:, x2.size(3) // 2:] = f_sub2[:, f_sub2.size(1) * 3 // 4:f_sub2.size(1) * 4 // 4, :,
                                                         :]
        x[:, 0:self.c_sub, :, :] = out1
        x[:, self.c_sub:, :, :] = out2
        return x


# 拉普拉斯梯度算子
class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        kernel = [[-1.0, -1.0, -1.0],
                  [-1.0, 8.0, -1.0],
                  [-1.0, -1.0, -1.0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand((3, 1, 3, 3))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.l1loss = nn.L1Loss()

    def forward(self, hr, sr):
        imglossh = self.l1loss(sr, hr)
        sr = F.conv2d(sr, self.weight, padding=1, groups=3)
        hr = F.conv2d(hr, self.weight, padding=1, groups=3)
        gradlossh = self.l1loss(sr, hr)
        return imglossh+gradlossh*0.001


# 罗布特梯度算子
class GradLossS(nn.Module):
    def __init__(self):
        super(GradLossS, self).__init__()
        kernelx = [[-1.0, 0.0, 1.0],
                  [-2.0, 0.0, 2.0],
                  [-1.0, 0.0, 1.0]]
        kernely = [[1.0, 2.0, 1.0],
                   [0.0, 0.0, 0.0],
                   [-1.0, -2.0, -1.0]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernelx = kernelx.expand((3, 1, 3, 3))
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        kernely = kernely.expand((3, 1, 3, 3))
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
        self.l1loss = nn.L1Loss()
        self.downsam = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=6, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, lr, hr, sr):
        b, c, h, w = lr.size()
        imglossh = self.l1loss(sr, hr)
        srx = F.conv2d(sr, self.weightx, padding=1, groups=3)
        hrx = F.conv2d(hr, self.weightx, padding=1, groups=3)
        sry = F.conv2d(sr, self.weighty, padding=1, groups=3)
        hry = F.conv2d(hr, self.weighty, padding=1, groups=3)
        sr = (srx ** 2 + sry ** 2 + 1e-5) ** 0.5
        hr = (hrx ** 2 + hry ** 2 + 1e-5) ** 0.5
        gradlossh = self.l1loss(sr, hr)
        return imglossh+0.01*gradlossh


# 求特征图梯度
class GradConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GradConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 0)
        self.r_u_pad = nn.ZeroPad2d(padding=(0, 2, 2, 0))
        self.l_d_pad = nn.ZeroPad2d(padding=(2, 0, 0, 2))

    def forward(self, x1, x2):
        x1 = self.r_u_pad(x1)
        x2 = self.l_d_pad(x2)
        x1 = self.conv(x1)
        x2 = self.conv(x2)

        sub_img = x1 - x2

        return sub_img


