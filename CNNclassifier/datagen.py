#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:53:56 2019

@author: illuminatus
"""
import numpy as np
from keras.utils import Sequence, to_categorical
import librosa
import pandas as pd


    
    

class DataGenerator(Sequence):
    
    
    def __init__(self, data_dir, params, list_IDs, labels=None, batch_size=32, 
                 shuffle=True, preproc_fn = lambda x: x):
        
        self.dim = params['dim']
        self.sampling_rate = params['sampling_rate']
        self.audio_duration = params['audio_duration']
        self.mfcc = params['mfcc']
        self.n_mfcc = params['n_mfcc']
        
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.n_classes = params['n_classes']
        self.shuffle = shuffle
        self.preproc_fn = preproc_fn
        self.on_epoch_end()


    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))
    


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)
    
    

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):
        # Initialization
        vbatch_size = len(list_IDs_temp)
        X = np.empty((vbatch_size, *self.dim))
        y = np.empty((vbatch_size), dtype=int)
        
        input_length = self.sampling_rate * self.audio_duration
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            fpath = self.data_dir + ID
            
            # Read and Resample the audio
            data, _ = librosa.core.load(fpath, sr=self.sampling_rate,
                                        res_type='kaiser_fast')
            
           # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
            
            
            if self.mfcc:
                data = librosa.feature.mfcc(data, sr=self.sampling_rate,
                                                   n_mfcc=self.n_mfcc)
                data = np.expand_dims(data, axis=-1)
                data = self.preproc_fn(data)
            else:
                data = self.preproc_fn(data)[:, np.newaxis]
            
            # Store sample
            X[i,] = data
         
        
        # Store class
        if self.labels is not None:
            y = np.empty(vbatch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            
            return X, to_categorical(y, num_classes=self.n_classes)
        
        else:
            return X
        
    
#==============================================================================




