#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:06:20 2019

@author: illuminatus
"""

import datagen
from CNNmodels import *
import pandas as pd
import numpy as np


from utils import *

#==============================================================================

def test_run1D(train, params, data_dir,num_epochs=2,restart=False):
    
    LABELS = list(train.label.unique())
    PREDICTION_FOLDER = "predictions_1d_conv"
    
    train_set = train.iloc[:550]
    val_set = train.iloc[550:]
    
    
    #-- Train the model -------
    train_gen = datagen.DataGenerator(data_dir,params, train_set.index,
                                      train_set.label_idx, batch_size=32, 
                                      shuffle = True, preproc_fn=normalize_audio
                                      )
    
    val_gen = datagen.DataGenerator(data_dir,params,val_set.index,
                                      val_set.label_idx, batch_size=32, 
                                      shuffle = False, preproc_fn=normalize_audio
                                      )
    
    #model = get_1d_dummy_model(mparams)
    model = model_1D_CNN_1(params)
    
    if restart == True:
        model.load_weights('CNN1_1.h5')
    
    history = model.fit_generator(train_gen, callbacks=None, 
                                  validation_data=val_gen,
                                  epochs=num_epochs, use_multiprocessing=True, 
                                  workers=4, max_queue_size=20)
    
    #--- Make train set predictions -------
    
    model.save_weights('CNN1_1.h5')
    
    train_gen = datagen.DataGenerator(data_dir, params, train.index,
                                               batch_size=128, 
                                      shuffle = False, preproc_fn=normalize_audio
                                      )
    print("Train Gen created.")
    predictions = model.predict_generator(train_gen, use_multiprocessing=True, 
                                          workers=4, max_queue_size=20, verbose=1)
    
    print("Train preds made successfully.",predictions.shape)
    np.save(PREDICTION_FOLDER + "/train_predictions_.npy", predictions)
    
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(str(x))) for x in top_3]
    train['pred_label'] = predicted_labels
    
    train['plabel'] = np.argmax(predictions,axis=1)
    return predictions
    
    
 

def test_run2D(train, params, data_dir, num_epochs=2, restart=False):
    
    LABELS = list(train.label.unique())
    PREDICTION_FOLDER = "predictions_2d_conv"
    
    train_set = train.iloc[:550]
    val_set = train.iloc[550:]
    

    #-- Train the model -------
    train_gen = datagen.DataGenerator(data_dir,params, train_set.index,
                                      train_set.label_idx, batch_size=32, 
                                      shuffle = False, preproc_fn=normalize_mfcc
                                      )
    
    val_gen = datagen.DataGenerator(data_dir,params,val_set.index,
                                      val_set.label_idx, batch_size=32, 
                                      shuffle = False, preproc_fn=normalize_mfcc
                                      )
    
    model = model_2D_MFCC(params)
    
    if restart == True:
        model.load_weights('CNN2_1.h5')
    
    history = model.fit_generator(train_gen, callbacks=None, 
                                  validation_data=val_gen,
                                  epochs=num_epochs, use_multiprocessing=True, 
                                  workers=4, max_queue_size=20)
    
    #--- Make train set predictions -------
    
    model.save_weights('CNN2_1.h5')
    
    train_gen = datagen.DataGenerator(data_dir, params, train.index,
                                               batch_size=128, 
                                      shuffle = False, preproc_fn=normalize_mfcc
                                      )
    print("Train Gen created.")
    predictions = model.predict_generator(train_gen, use_multiprocessing=True, 
                                          workers=4, max_queue_size=20, verbose=1)
    

    print("Train preds made successfully.",predictions.shape)
    np.save(PREDICTION_FOLDER + "/train_predictions_.npy", predictions)
    
    
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(str(x))) for x in top_3]
    train['pred_label'] = predicted_labels
    
    train['plabel'] = np.argmax(predictions,axis=1)
    
    return predictions
    
    
    
    
    
    
    
    
    
    
    
    
    
    