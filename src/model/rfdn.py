import torch
import torch.nn as nn
import block as B

def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=6, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.B5 = B.RFDB(in_channels=nf)
        self.B6 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        print(out_B6.shape)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import sys
    import numpy as np
    from thop import profile

    sys.path.append("../")
    from option import args
    args.scale = [2]
    # from option import args
    args.n_feats = 48
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RFDN().to(device)
    # 4倍 320/180  2倍  640/360  3倍  427/240
    input = torch.randn(1, 3, 640, 360).cuda()
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

    print("params:" , params_count(model))
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)