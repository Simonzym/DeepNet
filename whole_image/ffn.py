#build RNN for extracted features
import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import nibabel as nib
import random
import pickle
from itertools import chain
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

def build_ffn(lr = 0.001, alpha = 0.01,input_dim = 256):
    
    ffn_model = Sequential()
    ffn_model.add(Input(shape = (input_dim)))
    ffn_model.add(Dense(256, activation = 'relu', kernel_regularizer = l2(alpha)))
    ffn_model.add(Dropout(0.2))
    ffn_model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(alpha)))
    ffn_model.add(Dropout(0.2))
    ffn_model.add(Dense(1, activation = 'relu', kernel_regularizer = l2(alpha)))
    opt = Adam(lr = lr)
    ffn_model.compile(loss = 'mse', optimizer = opt, metrics = ['mse'])
    
    return ffn_model

def get_last(num_cv, cv_run, set_type = 'train'):
    
    path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/', set_type, '_nodes.csv'])
    graph_path = 'Code/Info/graphDIF12/graphs.csv'
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    for gid in all_gid:
        
        image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
        image_gid = image_gid[:, 0:644].astype('float32')
        num_img = image_gid.shape[0]     
        y_loc = list(status['graph_id']).index(gid)
        y.append(status['ADAS'][y_loc])
        seq.append(list(image_gid[num_img-1]))
        
    return np.array(seq), np.array(y)

for cv_run in range(1,6):
    for num_cv in range(1,6):
    

        train_seq, train_y = get_last(num_cv, cv_run, 'train')
        test_seq, test_y = get_last(num_cv, cv_run, 'test')
        
        metric = []
        img_metirc = []
        dif12_ffn = build_ffn(0.0005, 0.05, 640)

            
        dif12_fit = dif12_ffn.fit([train_seq[:,0:640]], train_y,
                                  batch_size = 32, epochs = 100, 
                                  validation_data = ([test_seq[:,0:640]], test_y))

        history = list(dif12_fit.history.values())
        history = np.transpose(np.array(history))
        # dif12_img_ffn = build_ffn(0.0005, 0.05, 256)
        # dif12_img_fit = dif12_img_ffn.fit([train_seq[:,0:256]], to_categorical(train_y),
        #                           batch_size = 32, epochs = 100, 
        #                           validation_data = ([test_seq[:,0:256]], to_categorical(test_y)))
        
        hist_df = pd.DataFrame(history, columns = ['loss', 'mse', 'val_loss', 'val_mse'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/ffn/adas.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_noimg_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/ffn/noimg_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_img_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/ffn/img_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)