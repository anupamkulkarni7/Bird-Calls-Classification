#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:05:35 2019

@author: illuminatus
"""

import numpy as np
from keras.models import Sequential
from keras.layers import (Conv1D, Conv2D, Dense, Dropout, GlobalMaxPooling1D,BatchNormalization,
                          GlobalMaxPool1D, Input, MaxPooling1D, MaxPooling2D, concatenate, Activation,
                          GlobalMaxPool2D, Convolution2D, MaxPool2D, Flatten)
from keras import losses, models, optimizers




def model_1D_CNN_1(params):
    
    n_classes = params['n_classes']
    input_len = params['audio_length']
    learning_rate = params['learning_rate']
    
    model = Sequential()
    
    #Conv-Conv-MaxPool(1): 32000 - 15996 - 
    model.add(Conv1D(filters = 16, kernel_size = 9, strides = 2 , padding = 'valid',
                     activation = 'relu',input_shape=(input_len,1)))
    
    model.add(Conv1D(filters = 24, kernel_size = 9, strides = 2 , padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling1D(4, padding = 'same'))
    model.add(Dropout(0.2))
    
    #Conv-Conv-MaxPool(1)
    model.add(Conv1D(filters = 32, kernel_size = 9, strides = 2 , padding = 'valid',
                     activation = 'relu'))
    model.add(Conv1D(filters = 32, kernel_size = 9, strides = 2 , padding = 'valid'))   
    model.add(BatchNormalization())
    model.add(Activation('relu'))
                     
    model.add(MaxPooling1D(4,padding = 'same'))
    model.add(Dropout(0.2))

    #Conv-Conv-MaxPool(1)
    model.add(Conv1D(filters = 64, kernel_size = 9, strides = 2 , padding = 'valid',
                     activation = 'relu'))
    model.add(Conv1D(filters = 128, kernel_size = 9, strides = 2 , padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))

    #FC Layers
    model.add(Dense(units = 128, kernel_initializer = 'glorot_uniform',activation = 'relu'))
    model.add(Dense(units = n_classes, kernel_initializer = 'glorot_uniform',activation = 'softmax'))
    
    opt = optimizers.Adam(learning_rate)
    model.compile(optimizer = opt, loss = losses.categorical_crossentropy, metrics=['acc'])
    
    return model
    

def model_2D_MFCC(params):
    
    n_classes = params['n_classes']
    dim = params['dim']
    learning_rate = params['learning_rate']
    
    model = Sequential()
    
    #--Block 1: conv-conv-maxpool
    model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'valid',
                     activation = 'relu',input_shape=(dim[0],dim[1],1)))
    model.add(Conv2D(filters = 24, kernel_size = (3,3), padding = 'valid'))
    model.add(BatchNormalization(momentum=0.01))
    model.add(Activation('relu'))
                     
    model.add(MaxPooling2D(pool_size=(2,3),padding = 'same'))
    model.add(Dropout(0.35))
    
    #--Block 2: conv-conv-maxpool
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid',
                     activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid'))
    model.add(BatchNormalization(momentum=0.01))
    model.add(Activation('relu'))
                     
    model.add(MaxPooling2D(pool_size=(2,3),padding = 'same'))
    model.add(Dropout(0.35))
    
    #--Block 3: conv-conv-maxpool
    model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'valid',
                     activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'valid'))
    model.add(BatchNormalization(momentum=0.01))
    model.add(Activation('relu'))
                     
    model.add(MaxPooling2D(pool_size=(3,5),padding = 'same'))
    model.add(Dropout(0.35))
    model.add(Flatten())
    
    #Output Layer
    model.add(Dense(units = n_classes, kernel_initializer = 'glorot_uniform',
                    activation = 'softmax'))
    
    opt = optimizers.Adam(learning_rate)
    model.compile(optimizer = opt, loss = losses.categorical_crossentropy, metrics=['acc'])
    
    return model


    
def get_2d_conv_model(params):
    
    nclass = params['n_classes']
    dim = params['dim']
    learning_rate = params['learning_rate']
    
    inp = Input(shape=(dim[0],dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model



