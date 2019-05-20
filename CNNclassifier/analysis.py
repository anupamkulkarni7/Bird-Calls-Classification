#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:44:06 2019

@author: illuminatus
"""

import numpy as np
np.random.seed(1001)

import os
import shutil

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import librosa
import wave


matplotlib.style.use('ggplot')

"""
train = pd.read_csv("freesound-audio-tagging/train.csv")
test = pd.read_csv("freesound-audio-tagging/sample_submission.csv")

train.head()

print("Number of examples",train.shape[0],"Number of classes",len(train.label.unique()))

import IPython.display as ipd  # To play sound in the notebook
fname = 'freesound-audio-tagging/audio_train/' + '00044347.wav'   # Hi-hat
ipd.Audio(fname)

SAMPLE_RATE = 44100
wav, _ = librosa.core.load(fname, sr=SAMPLE_RATE)
wav = wav[:2*44100]
mfcc = librosa.feature.mfcc(wav, sr = SAMPLE_RATE, n_mfcc=40)
plt.imshow(mfcc, cmap='hot', interpolation='nearest');


import wave
wav = wave.open(fname)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())

from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)
plt.plot(data)


train['nframes'] = train['fname'].apply(lambda f: wave.open('freesound-audio-tagging/audio_train/' + f).getnframes())
test['nframes'] = test['fname'].apply(lambda f: wave.open('freesound-audio-tagging/audio_test/' + f).getnframes())

_, ax = plt.subplots(figsize=(16, 4))
sns.violinplot(ax=ax, x="label", y="nframes", data=train)
plt.xticks(rotation=90)
plt.title('Distribution of audio frames, per label', fontsize=16)
plt.show()
"""


train = pd.read_csv("../syllables.csv")
train.head()
print("Number of examples",train.shape[0],"Number of classes",len(train.label.unique()))

fname = '../syllables/' + '1014_0_0.wav'
#wave_file = wave.open(fname, "rb")
#frame_rate = wave_file.getframerate()

SAMPLE_RATE = 48000
wav, _ = librosa.core.load(fname, sr=SAMPLE_RATE)
wav = wav[:32000]

stft = librosa.core.stft(wav)
stft = np.abs(stft)
#plt.imshow(stft, cmap='hot', interpolation='nearest')

mfcc = librosa.feature.mfcc(wav, sr=SAMPLE_RATE, n_mfcc=40)
plt.imshow(mfcc, cmap='hot', interpolation='nearest')



train['nframes'] = train['fname'].apply(lambda f: wave.open('../syllables/' + f).getnframes())
#train['duration'] = train['nframes'].apply(lambda nf: frame_rate)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
train.nframes.hist(bins=100, ax=axes[0])
train.nframes.hist(bins=100, ax=axes[1])
plt.suptitle('Frame Length Distribution in Train and Test', ha='center', fontsize='large');



import IPython.display as ipd
fname = 'syllables/997_2_14.wav'
ipd.Audio(fname)


#------------------------------------------------------------------------------





