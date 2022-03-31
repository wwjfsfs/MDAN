import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import os
import random
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch.manual_seed(args.seed)
seed = args.seed
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
checkpoint = utility.checkpoint(args)
random.seed(seed)
np.random.seed(seed)

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            print("params:", params_count(_model))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


if __name__ == '__main__':
    main()
