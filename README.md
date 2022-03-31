python 3.6
torch 1.1.0

validation set(DIV2K 801-900): /media/lss/Newsmy/EDSR-PyTorch-master/dataset/benchmark/VAL

#train x2
python main.py --model mdan --scale 2 --patch_size 128 --batch_size 16 --save mdan_x2

#train x4
python main.py --model mdan --scale 4 --patch_size 256 --batch_size 16 --save mdan_x4 --pre_train /home/lss/workspace/EDSR-PyTorch-master/experiment/mdan_x2/model_latest.pt 

