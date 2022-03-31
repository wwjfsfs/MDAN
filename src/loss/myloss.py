import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient(y):
    gradient_h = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    return gradient_h, gradient_y


class MYLOSS(nn.Module):
    def __init__(self, args):
        super(MYLOSS, self).__init__()
        self.l1loss = nn.L1Loss()

    def forword(self, shalow_sr, hr):
        # l1_loss = self.l1loss(sr, hr)
        grad_hr_h, grad_hr_w = gradient(hr)
        grad_sr_h, grad_sr_w = gradient(shalow_sr)
        loss_h = self.l1loss(grad_hr_h, grad_sr_h)
        loss_w = self.l1loss(grad_hr_w, grad_sr_w)
        loss = 0.001 * (loss_h + loss_w)
        return loss








criterionCAE = nn.L1Loss()
criterionCAE.cude()


lOSS_H = criterionCAE()
LOSS_Y = criterionCAE()