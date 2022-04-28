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
#outcome
def brier(y_true, y_pred):
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis = 1))/2


def build_ffn(alpha = 0.01, lr = 0.001, input_dim = 256):
    
    ffn_model = Sequential()
    ffn_model.add(Input(shape = (input_dim)))
    ffn_model.add(Dense(256, activation = 'relu', kernel_regularizer = l2(alpha)))
    ffn_model.add(Dropout(0.2))
    ffn_model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(alpha)))
    ffn_model.add(Dropout(0.2))
    ffn_model.add(Dense(2, activation = 'softmax'))
    opt = Adam(lr = lr)
    ffn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', 'AUC'])
    
    return ffn_model

def get_last(num_cv, cv_run, set_type = 'train', missing = 1):
    
    path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/', set_type, '_nodes.csv'])
    graph_path = 'Code/Info/graphDIF12/graphs.csv'
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    for gid in all_gid:
        
        y_loc = list(status['graph_id']).index(gid)
        label = status['label'][y_loc]
        if label != missing:
            
            image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
            image_gid = image_gid[:, 0:644].astype('float32')
            num_img = image_gid.shape[0]     
            y.append(label)
            seq.append(list(image_gid[num_img-1]))
        
    return np.array(seq), np.array(y)

for cv_run in range(1,6):
    for num_cv in range(1,6):
    

        #NC vs. AD
        train_seq, train_y = get_last(num_cv, cv_run, 'train', 1)
        test_seq, test_y = get_last(num_cv, cv_run, 'test', 1)
        
        ncad_ffn = build_ffn(0.0002, 0.005, 643)
        metric = []
        img_metirc = []
        for ep in range(1, 101):
            
            ncad_fit = ncad_ffn.fit([train_seq[:,0:643]], to_categorical(train_y/2),
                                      batch_size = 32, epochs = 1, 
                                      validation_data = ([test_seq[:,0:643]], to_categorical(test_y/2)), verbose = 2)
            preds = ncad_ffn.predict(test_seq[:,0:643])
            brier_score = brier(to_categorical(test_y/2), preds)
            #auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(ncad_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            metric.append(history)
        
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'auc', 'val_loss', 'val_accuracy','val_auc', 'brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/ffn/ncad_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        #NC vs. MCI
        train_seq, train_y = get_last(num_cv, cv_run, 'train', 2)
        test_seq, test_y = get_last(num_cv, cv_run, 'test', 2)
        
        ncmci_ffn = build_ffn(0.0002, 0.005, 643)
        metric = []
        img_metirc = []
        for ep in range(1, 101):
            
            ncmci_fit = ncmci_ffn.fit([train_seq[:,0:643]], to_categorical(train_y),
                                      batch_size = 32, epochs = 1, 
                                      validation_data = ([test_seq[:,0:643]], to_categorical(test_y)), verbose = 2)
            preds = ncmci_ffn.predict(test_seq[:,0:643])
            brier_score = brier(to_categorical(test_y), preds)
            #auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(ncmci_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            metric.append(history)
        
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'auc', 'val_loss', 'val_accuracy','val_auc', 'brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/ffn/ncmci_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        #MCI vs. AD
        train_seq, train_y = get_last(num_cv, cv_run, 'train', 0)
        test_seq, test_y = get_last(num_cv, cv_run, 'test', 0)
        
        mciad_ffn = build_ffn(0.0002, 0.005, 643)
        metric = []
        img_metirc = []
        for ep in range(1, 101):
            
            mciad_fit = mciad_ffn.fit([train_seq[:,0:643]], to_categorical(train_y-1),
                                      batch_size = 32, epochs = 1, 
                                      validation_data = ([test_seq[:,0:643]], to_categorical(test_y-1)), verbose = 2)
            preds = mciad_ffn.predict(test_seq[:,0:643])
            brier_score = brier(to_categorical(test_y-1), preds)
            #auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(mciad_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            metric.append(history)
        
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'auc', 'val_loss', 'val_accuracy','val_auc', 'brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/ffn/mciad_all.csv'])
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