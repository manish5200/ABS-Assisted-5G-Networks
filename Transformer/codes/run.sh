python train.py --data_name=foursquare --data_path=../data/

colab -> 

!unzip Transformer.zip
%ls

import os
os.chdir('Transformer/codes/')
print(os.getcwd())
%ls

#python train.py --data_name=foursquare --data_path=../data/
# coding: utf-8
from __future__ import print_function
from __future__ import division
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
import json
import time
import argparse
import numpy as np
from json import encoder

from train import run
seed = int(time.time())
torch.manual_seed(seed)
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = argparse.Namespace(
    loc_emb_size=512,
    uid_emb_size=128,
    tim_emb_size=16,
    dropout_p=0.1,
    data_name='foursquare',
    learning_rate=5 * 1e-5,
    lr_step=3,
    lr_decay=0.1,
    optim='Adam',
    L2=1 * 1e-5,
    clip=5.0,
    epoch_max=50,
    data_path='../data/',
    save_path='../results/',
    pretrain=0
)

ours_acc = run(args)
print('ours_acc:' + str(ours_acc))