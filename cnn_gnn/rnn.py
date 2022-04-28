#build RNN for extracted features
import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import nibabel as nib
import random
import pickle
import sklearn
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

#build model
def build_rnn(lr = 0.001, alpha = 0.02, size = 258, num_classes = 2):
    
    rnn_model = Sequential()
    rnn_model.add(Masking(mask_value=-1, input_shape = (9, size)))
    rnn_model.add(Bidirectional(GRU(256, return_sequences=True, kernel_regularizer = l2(alpha))))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Bidirectional(GRU(128, kernel_regularizer = l2(alpha))))
    rnn_model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(alpha)))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Dense(num_classes, activation = 'softmax', kernel_regularizer = l2(alpha)))
    opt = Adam(lr = lr)
    rnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy', 'AUC'])
    return rnn_model


def convert_seq(num_cv, cv_run, set_type = 'train', missing = 1):
    
    path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv),  '/run', str(cv_run), '/', set_type, '_nodes.csv'])
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
            if num_img < 9:
                sup_img = np.zeros((9 - num_img, 644)) - 1
                image_gid = np.vstack([image_gid, sup_img])
            
            y.append(label)
            seq.append(list(image_gid))
        
    return np.array(seq), np.array(y)

for cv_run in range(1,6):
    for num_cv in range(5):
        j = num_cv + 1

        
        #NC vs. AD
        train_seq, train_y = convert_seq(j, cv_run, 'train', 1)
        test_seq, test_y = convert_seq(j, cv_run, 'test', 1)
        
        ncad_rnn = build_rnn(0.0002, 0.005, 643)
        metric = []
        img_metirc = []
        for ep in range(1, 151):
            
            ncad_fit = ncad_rnn.fit([train_seq[:,:,0:643]], to_categorical(train_y/2),
                                      batch_size = 32, epochs = 1, 
                                      validation_data = ([test_seq[:,:,0:643]], to_categorical(test_y/2)), verbose = 2)
            preds = ncad_rnn.predict(test_seq[:,:,0:643])
            brier_score = brier(to_categorical(test_y/2), preds)
            #auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(ncad_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            metric.append(history)
        
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'auc', 'val_loss', 'val_accuracy','val_auc', 'brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/rnn/ncad_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        #NC vs. MCI
        train_seq, train_y = convert_seq(j, cv_run, 'train', 2)
        test_seq, test_y = convert_seq(j, cv_run, 'test', 2)
        
        ncmci_rnn = build_rnn(0.0002, 0.005, 643)
        metric = []
        img_metirc = []
        for ep in range(1, 151):
            
            ncmci_fit = ncmci_rnn.fit([train_seq[:,:,0:643]], to_categorical(train_y),
                                      batch_size = 32, epochs = 1, 
                                      validation_data = ([test_seq[:,:,0:643]], to_categorical(test_y)), verbose = 2)
            preds = ncmci_rnn.predict(test_seq[:,:,0:643])
            brier_score = brier(to_categorical(test_y), preds)
            #auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(ncmci_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            metric.append(history)
        
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'auc', 'val_loss', 'val_accuracy','val_auc', 'brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/rnn/ncmci_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        #MCI vs. AD
        train_seq, train_y = convert_seq(j, cv_run, 'train', 0)
        test_seq, test_y = convert_seq(j, cv_run, 'test', 0)
        
        mciad_rnn = build_rnn(0.0002, 0.005, 643)
        metric = []
        img_metirc = []
        for ep in range(1, 151):
            
            mciad_fit = mciad_rnn.fit([train_seq[:,:,0:643]], to_categorical(train_y-1),
                                      batch_size = 32, epochs = 1, 
                                      validation_data = ([test_seq[:,:,0:643]], to_categorical(test_y-1)), verbose = 2)
            preds = mciad_rnn.predict(test_seq[:,:,0:643])
            brier_score = brier(to_categorical(test_y-1), preds)
            #auc = roc_auc_score(to_categorical(test_y), preds, multi_class = 'ovo')
            history = list(mciad_fit.history.values())
            history = list(chain(*history))
            history.append(brier_score)
            metric.append(history)
        
        hist_df = pd.DataFrame(metric, columns = ['loss', 'Accuracy', 'auc', 'val_loss', 'val_accuracy','val_auc', 'brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/rnn/mciad_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            

        
     

            

        # dif12_noimg_rnn = build_rnn(0.0005, 0.05, 3)
        # dif12_noimg_fit = dif12_noimg_rnn.fit([train_seq[:,:,256:259]], to_categorical(train_y),
        #                           batch_size = 32, epochs = 100, 
        #                           validation_data = ([test_seq[:,:,256:259]], to_categorical(test_y)))
        
        # for ep in range(1, 100):
        #     dif12_img_fit = dif12_img_rnn.fit([train_seq[:,:,0:256]], to_categorical(train_y),
        #                               batch_size = 32, epochs = 1, 
        #                               validation_data = ([test_seq[:,:,0:256]], to_categorical(test_y)), verbose = 2)
            

            
        # hist_df = pd.DataFrame(dif12_noimg_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/rnn_noimg_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)
            
        # hist_df = pd.DataFrame(dif12_img_fit.history)
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/rnn/diag_img_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)