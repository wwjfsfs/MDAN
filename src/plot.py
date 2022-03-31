import numpy as np
from option import args
import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from matplotlib import cm
import matplotlib.ticker as ticker
# plt.style.use('ggplot')
import imageio
from data import common
from importlib import import_module
import model
import utility
# import seaborn as sns
# sns.set()
import random
from PIL import Image
from torchvision import transforms
import pandas as pd
# import pylab as pl.

np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activation_out = {}
activation_in = {}

def get_activation(name):
    def hook(model, input, output):
        activation_out[name] = output.detach()
    return hook

def load_file(lr_p, hr_p=None):  # 图片预处理，获得lr和hr中小块图片
    # hr = imageio.imread(lr_p)
    lr = imageio.imread(lr_p)
    # scale = 1
    # ih, iw = lr.shape[:2]
    # hr = hr[0:ih * scale, 0:iw * scale]
    # lr = common.set_channel(lr, n_channels=args.n_colors)[0]  # n_colors默认为3
    # lr = np.array(lr)
    pair_t = common.np2Tensor(lr, rgb_range=args.rgb_range)

    return pair_t[0]

def load_model(args, apath, model=None):
    module = import_module('model.' + args.model.lower())  # .lower()所有大写转小写,动态导入模块（默认edsr.py）
    model = module.make_model(args).to(device)

    load_from = None
    kwargs = {}
    load_from = torch.load(
        os.path.join(apath, 'model_999.pt'),
        **kwargs
    )
    print(os.path.join(apath, 'model_999.pt'))
    model.load_state_dict(load_from, strict=True)

    return model

# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        # outputs.append(x.squeeze(0))
        # print('---------', self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            # print(module)
            x = module(x)
            print('name', name)

            if name == 'upsample':
                for i in range(16):
                    m = module[i]
                    outputs.append(m(x))
                break
            # if name in self.extracted_layers:
            #     # continue
            #     outputs.append(x)
            #     break

        return outputs
def get_feature(model):  # 特征可视化
    # 输入数据
    config = {}
    img_p = '/home/zzr/EDSRpp/benchmark/Set14/LR_bicubic/X3/zebrax3.png'
    lr = load_file(img_p)
    lr = lr.unsqueeze(0)  # 【1,3,224,224】
    lr = lr.to(device)

    # 特征输出
    # net = model().to(device)
    net = model
    # net.load_state_dict(torch.load('./model/net_050.pth'))
    exact_list = ['body']
    myexactor = FeatureExtractor(net, exact_list)  # 输出是一个网络
    sr = myexactor(lr)
    return sr


def get_feature_visualization(sr_feature, name='', mode=None):
    sr_feature = sr_feature[0]
    # 特征输出可视化
    # if index is not None:
    #     candidate = index

    if mode == 'avg':
        sr_feature = sr_feature.mean(1)
        plt.imshow(sr_feature.data.cpu().numpy(), cmap='jet')  # 红色响应值高
        plt.savefig('test_%s.png' % name, dpi=500, pad_iches=0)
        plt.axis('off')
        plt.show()  #
    elif mode == 'norm':
        mean = sr_feature.mean(1, keepdim=True)
        var = sr_feature.var(1, keepdim=True)
        sr_feature = (sr_feature - mean) / torch.sqrt(var + 1e-6)
        plt.imshow(sr_feature.data.cpu().numpy(), cmap='jet')  # 红色响应值高
        plt.savefig('test_%s.png' % name, dpi=500, pad_iches=0)
        plt.show()  #
    else:
        #ax = plt.subplots(8, 8, constrained_layout=True)
        #fig = plt.figure(figsize=(4, 4))
        # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        plt.figure(facecolor=[0.5, 0.5, 0.5])
        for i in range(24):  # 可视化了64通道
            ax = plt.subplot(4, 6, i + 1)
            plt.axis('off')
            plt.imshow(normalization(sr_feature[i, :, :]).data.cpu().numpy(), cmap='jet')  # 红色响应值高

        ''''''
        #plt.subplots_adjust(left=0.10, top=0.88, right=0.90, bottom=0.08, wspace=0.01, hspace=0.01)
        plt.subplots_adjust(left=0.08, top=1.0, right=1.5, bottom=0.05, wspace=0.001, hspace=0.001)
        #plt.subplots_adjust(left=0, top=0.2, right=0.6, bottom=0, wspace=0, hspace=0)  #2 8  16张图片的布局
        #cax = plt.axes([0.62, 0.01, 0.005, 0.18]) # left dis,       bar weight, bar height

        cax = plt.axes([1.54, 0.05, 0.025, 0.95]) # left dis,       bar weight, bar height
        cb=plt.colorbar(cax=cax)
        cb.ax.tick_params(labelsize=20)
        #plt.colorbar()
        plt.savefig('test_%s_avg.png' % name, dpi=300, bbox_inches='tight')
        plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型

def normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


# def get_weights(model):
#     ans = 0
#     model_dict = model.state_dict()
#     for k, v in model_dict.items():
#         if k == 'upsample.cb.embedding_out.0':
#             ans = v
#             sig_ans = nn.Sigmoid()(v)
#
#     return ans.detach().cpu().numpy(), sig_ans.detach().cpu().numpy()


if __name__ == "__main__":
    import scipy.io as sio
    root = r'/home/lss/workspace/EDSR-PyTorch-master/experiment/upsrb_2x/model/'   #MDCB和上采样均用CB注意力，有imbedding的CB注意力 137MDCN_SCP_DIV5 125MDCN_nomodule 124MDCN_MDCB_CB
    # path = r'model_best.pt'
    args.model = 'UPSRB'
    args.scale = [2,3,4]
    model = load_model(args, root)
    # print(model)
    # model = model.cuda()
    #model.eval()

    # sr_feature_c = get_feature(model)
    #layer_name = 'upsample.pro_up.0.body.3'

    layer_name = 'BS1.bs1_1'

    # layer_name = 'body.11.confusion_bottle'
    #layer_name = 'body.11.relu'
    for name, m in model.named_modules():
        if name == layer_name:
            m.register_forward_hook(get_activation(layer_name))


    img_p = '/home/lss/workspace/EDSR-PyTorch-master/dataset/benchmark/Set5/LR_bicubic/X2/birdx2.png'
    lr = load_file(img_p)
    lr = lr.unsqueeze(0)  #
    lr = lr.to(device)
    model(lr)
    sr_feature = activation_out[layer_name]
    sio.savemat("mdcn.mat",{'feat_map':sr_feature.detach().cpu().numpy()[0,0:24, :, :]})
    mid = sr_feature.detach().cpu().numpy()[0,0:24, :, :]
    print(mid.shape)
    get_feature_visualization(sr_feature, name=layer_name, mode='')
    print('finish!')
    '''
    
    # Set up plot
    def noramlization(data):
        minVals = data.min()
        maxVals = data.max()
        ranges = maxVals - minVals
        normData = np.zeros(np.shape(data))
        m = data.shape[0]
        normData = data - minVals
        normData = normData / ranges
        return normData

    dist_matrix = channel_dist.cpu().detach().numpy()
    # dist_matrix = np.around(dist_matrix, decimals=2)
    normData = noramlization(dist_matrix)
    fig = plt.figure()
    # figsize = (24, 16)
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(dist_matrix, annot=False, cmap='jet')
    # plt.imshow(dist_matrix, cmap='jet')
    ax.invert_yaxis()
    plt.xticks(rotation=0)
    # plt.title('BN-0-32')
    # plt.title('RN-0-32')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.show()
    fig.savefig('dist_matrix.pdf')
    '''
