#build Transformer for extracted features
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
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu


#outcome
def brier(y_true, y_pred):
    return K.mean(K.sum(K.pow(y_true - y_pred, 2), axis = 1))/2


class TransformerBlock(Layer):
    def __init__(self, feats_dim, num_heads, ff_dim, rate=0.1, alpha = 0.01):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feats_dim,
                                             kernel_regularizer = l2(alpha))
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu", kernel_regularizer = l2(alpha)), 
             Dense(feats_dim, kernel_regularizer = l2(alpha)),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        # self.query = Dense(9, activation = 'linear')
        # self.key = Dense(9, activation = 'linear')
        # self.value = Dense(9, activation = 'linear')

    def call(self, inputs):
        # query = tf.transpose(self.query(tf.transpose(inputs, perm = [0,2,1])), perm = [0,2,1])
        # key = tf.transpose(self.key(tf.transpose(inputs, perm = [0,2,1])), perm = [0,2,1])
        # value = tf.transpose(self.value(tf.transpose(inputs, perm = [0,2,1])), perm = [0,2,1])
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
def build_trans(inputs_shape, ffn_dim, alpha = 0.01, lr = 0.001, rate = 0.1):
    
    inputs = Input(shape = inputs_shape)
    trans_block1 = TransformerBlock(inputs_shape[1], 3, ffn_dim, alpha = alpha, rate = rate)
    trans_block2 = TransformerBlock(inputs_shape[1], 3, 128, alpha = alpha, rate = rate)
    #trans_block3 = TransformerBlock(inputs_shape[1], 3, ffn_dim, alpha = alpha, rate = rate)
    #trans_block4 = TransformerBlock(inputs_shape[1], 3, ffn_dim, alpha = alpha, rate = rate)
    x = trans_block1(inputs)
    x = trans_block2(x)
    #x = trans_block3(x)
    #x = trans_block4(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = Dropout(rate)(x)
    x = Dense(128, activation = 'relu', kernel_regularizer = l2(alpha))(x)
    x = Dropout(rate)(x)
    outputs = Dense(2, activation = 'softmax')(x)
    
    trans_model = Model(inputs = inputs, outputs = outputs)
    opt = Adam(lr = lr)
    trans_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy' ,'AUC', brier])
    
    return trans_model
    
_, _, _, _, _, _, _, visits_info = get_data('BP')
#take the extracted nodes for each graph, and build sequences
def trans_seq(num_cv, cv_run, set_type = 'train', missing = 1, start = 0, end = 643):
    
    path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/', 'run', str(cv_run), '/' ,set_type, '_nodes.csv'])
    graph_path = 'Code/Info/graphDIF12/graphs.csv'
    
    dataset = pd.read_csv(path)
    status = pd.read_csv(graph_path)
    all_gid = list(set(dataset['graph_id']))
    seq = []
    y = []
    num_seq = []
    
    d_model = end - start
    
    for gid in all_gid:
        
        y_loc = list(status['graph_id']).index(gid)
        label = status['label'][y_loc]
        
        if label != missing:
            
            times = visits_info[gid]
            image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
            image_gid = image_gid[:, start:end].astype('float32')
            num_img = image_gid.shape[0]
            num_seq.append(num_img)
            #calculate pe (positional embedding) for each sequence       
            pe = np.zeros((num_img, d_model))
            position = np.expand_dims(times/6, axis = 1)
            div_term_1 = np.exp(np.arange(0, d_model, 2) * -np.log(10000)/d_model)
            div_term_2 = np.exp((np.arange(1, d_model, 2)-1) * -np.log(10000)/d_model)
            pe[:, 0::2] = np.sin(position * div_term_1)
            pe[:, 1::2] = np.cos(position * div_term_2)
            #features + positional embedding
            image_pe_gid = image_gid + pe
            #exp
            if num_img < 9:
                sup_img = np.zeros((9 - num_img, d_model))
                image_pe_gid = np.vstack([image_pe_gid, sup_img])
                   
            y.append(label)
            seq.append(list(image_pe_gid))
        
    return np.array(seq), np.array(y)
    
for cv_run in range(1,6):
    for num_cv in range(5):
    
        j = num_cv + 1

        #NC vs. AD
        train_seq, train_y = trans_seq(j, cv_run, 'train')
        test_seq, test_y = trans_seq(j, cv_run, 'test')
        
        
        ncad_trans = build_trans(train_seq[0].shape, 256, alpha = 0.005, lr = 0.0002, rate = 0.2)

        metric = []
        img_metirc = []

            
        ncad_fit = ncad_trans.fit([train_seq[:,:,0:643]], to_categorical(train_y/2),
                                  batch_size = 32, epochs = 100,  shuffle = True,
                                  validation_data = ([test_seq[:,:,0:643]], to_categorical(test_y/2)), verbose = 2)     

        history = list(ncad_fit.history.values())
        history = np.transpose(np.array(history))

        hist_df = pd.DataFrame(history, columns = ['loss', 'Accuracy', 'auc', 'brier', 'val_loss', 'val_accuracy','val_auc', 'val_brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/trans/ncad_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        #NC vs. MCI
        train_seq, train_y = trans_seq(j, cv_run, 'train', 2)
        test_seq, test_y = trans_seq(j, cv_run, 'test', 2)
        
        
        ncmci_trans = build_trans(train_seq[0].shape, 256, alpha = 0.005, lr = 0.0002, rate = 0.2)


        metric = []
        img_metirc = []

            
        ncmci_fit = ncmci_trans.fit([train_seq[:,:,0:643]], to_categorical(train_y),
                                  batch_size = 32, epochs = 100,  shuffle = True,
                                  validation_data = ([test_seq[:,:,0:643]], to_categorical(test_y)), verbose = 2)     

        history = list(ncmci_fit.history.values())
        history = np.transpose(np.array(history))

        hist_df = pd.DataFrame(history, columns = ['loss', 'Accuracy', 'auc', 'brier', 'val_loss', 'val_accuracy','val_auc', 'val_brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/trans/ncmci_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

            
        #MCI vs. AD
        train_seq, train_y = trans_seq(j, cv_run, 'train', 0)
        test_seq, test_y = trans_seq(j, cv_run, 'test', 0)
        
        
        mciad_trans = build_trans(train_seq[0].shape, 256, alpha = 0.005, lr = 0.0002, rate = 0.2)

        metric = []
        img_metirc = []

            
        mciad_fit = mciad_trans.fit([train_seq[:,:,0:643]], to_categorical(train_y-1),
                                  batch_size = 32, epochs = 100,  shuffle = True,
                                  validation_data = ([test_seq[:,:,0:643]], to_categorical(test_y-1)), verbose = 2)     

        history = list(mciad_fit.history.values())
        history = np.transpose(np.array(history))

        hist_df = pd.DataFrame(history, columns = ['loss', 'Accuracy', 'auc', 'brier', 'val_loss', 'val_accuracy','val_auc', 'val_brier'])
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(j), '/run', str(cv_run), '/trans/mciad_all.csv'])
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

            
