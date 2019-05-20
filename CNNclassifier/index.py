#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:01:32 2019

@author: illuminatus
"""

import numpy as np
np.random.seed(1001)
import pdb

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import wave

from runs import *
from utils import *


DATA_DIR = '../syllables/'
DATA_CSV = '../syllables.csv'

#Read in data, and set things up
train = pd.read_csv(DATA_CSV)
train = train.sample(frac=1).reset_index(drop=True)

train['nframes'] = train['fname'].apply(lambda f: wave.open(DATA_DIR + f).getnframes())
num_lbls = list(train.label.unique())
lbl2indx = {lbl: i for i, lbl in enumerate(num_lbls)}
train.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: lbl2indx[x])


#-------------------------

"""
params = set_params(n_classes = 39, learning_rate = 0.0001, sampling_rate = 16000,
                    audio_duration = 2, n_mfcc=40,mfcc=False)

p = test_run1D(train,params,DATA_DIR,num_epochs=1,restart=True)
"""



params = set_params(n_classes = 39, learning_rate = 0.0001, sampling_rate = 16000,
                    audio_duration = 2, n_mfcc=40, mfcc=True)

p = test_run2D(train,params,DATA_DIR,num_epochs=10,restart=True)










