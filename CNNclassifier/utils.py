#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:33:17 2019

@author: illuminatus
"""
import numpy as np
import pandas as pd

def normalize_audio(x):
    max_ = np.max(x)
    min_ = np.min(x)
    x = (x - min_)/(max_ - min_ + 1e-8)
    return x - 0.5

def normalize_mfcc(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean)/std
    return X

def set_params(n_classes, learning_rate = 0.0001, sampling_rate = 16000,audio_duration = 2,
               n_mfcc=20,mfcc=False):
    
    params = {}
    params['n_mfcc'] = n_mfcc
    params['mfcc'] = mfcc
    params['audio_duration'] = audio_duration
    params['sampling_rate'] = sampling_rate
    
    audio_length = audio_duration*sampling_rate
    params['audio_length'] = audio_length
    
    if mfcc:
        dim = (n_mfcc, 1 + int(np.floor(audio_length/512)), 1)
    else:
        dim = (audio_length, 1)
    
    params['dim'] = dim
    params['n_classes'] = n_classes
    params['learning_rate'] = learning_rate

    return params


    
    