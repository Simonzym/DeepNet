#one CNN for feature extraction
import tensorflow as tf 
import numpy as np
import pandas as pd
import nibabel as nib
import random
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Reshape, BatchNormalization as BN
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Masking, Concatenate, Activation, Input, Layer
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.activations import relu
from sklearn.model_selection import StratifiedKFold
import logging


ncad, ncad_diag, images, diags, scores, diags_y, scores_y, visits_info = get_data('PET')

#split train and test according number of samples (not number of images)
all_gid = list(images.keys())
outcomes = list(diags_y.values())

n = len(all_gid)

#create 5-fold CV (index) (do not run this for a second time)
num_folds = 5
# i = 0
# kfold = StratifiedKFold(n_splits=num_folds, shuffle = True)

# #save training gid and test gid
# for t1, t2 in kfold.split(np.zeros(n), outcomes):
    
#     i = i+1
#     train_indexes = pd.DataFrame({'train': t1})
#     test_indexes = pd.DataFrame({'test': t2})
    
#     hist_csv_file = ''.join(['Code/Info/graphDIF12/cv',str(i), '/train.csv'])
#     with open(hist_csv_file, mode='w') as f:
#         train_indexes.to_csv(f)
        
#     hist_csv_file = ''.join(['Code/Info/graphDIF12/cv',str(i), '/test.csv'])
#     with open(hist_csv_file, mode='w') as f:
#         test_indexes.to_csv(f)
    
#run this for different cv by changing the value of i
#i should be change through 1 to 5

#for each cv, run 5 times
for cv_run in range(1, 6):
    
    for j in range(5):
        
        num_cv = j + 1
        train_index = pd.read_csv(''.join(['Code/Info/graphDIF12/cv',str(num_cv), '/train.csv']))
        test_index = pd.read_csv(''.join(['Code/Info/graphDIF12/cv',str(num_cv), '/test.csv']))
        
        train_index = list(train_index['x'])
        test_index = list(test_index['x'])
         
        train_images = []
        test_images = []
        
        train_diags = []
        test_diags = []
        
        train_scores = []
        test_scores = []
        
        
        for gid in train_index:        

            train_images = train_images + list(ncad[gid])
            train_diags = train_diags + list(ncad_diag[gid])
        
        for gid in test_index:

            test_images = test_images + list(ncad[gid]) 
            test_diags = test_diags + list(ncad_diag[gid])
        
        
        #to np array
        train_images = np.array(train_images)
        test_images = np.array(test_images)

        
        
        train_diags = np.array(train_diags)

        test_diags = np.array(test_diags)

        

        seq_diags = to_categorical(train_diags/2)
        
        
        seq_test_diags = to_categorical(test_diags/2)
        
        class CNNBlock(Layer):
            def __init__(self, input_dim = 91*109*91, name = None, alpha = 0.01):
                super(CNNBlock, self).__init__(name = name)
                self.l1 = Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer = l2(alpha), padding = 'same')
                self.l2 = MaxPooling3D(pool_size=(2, 2, 2), padding = 'same')
                self.l3 = BN()
                self.l4 = Dropout(0.4)
                self.l5 = Conv3D(16, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha), padding = 'same')
                self.l6 = MaxPooling3D(pool_size = (2,2,2), padding = 'same')
                self.l7 = BN()
                self.l8 = Dropout(0.4)
                self.l9 = Conv3D(32, kernel_size=(3,3,3), activation = 'relu', kernel_regularizer = l2(alpha), padding = 'same')
                self.l10 = MaxPooling3D(pool_size = (2,2,2), padding = 'same')
                self.l11 = BN()
                self.l12 = Dropout(0.4)
                self.l13 = Flatten()
        
            def call(self, inputs):
                x = self.l1(inputs)
                x = self.l2(x)
                x = self.l3(x)
                x = self.l4(x)
                x = self.l5(x)
                x = self.l6(x)
                x = self.l7(x)
                x = self.l8(x)
                x = self.l9(x)
                x = self.l10(x)
                x = self.l11(x)
                x = self.l12(x)
                x = self.l13(x)
        
                return x
                
        
        def mymodel(input_shape = (91,109,91,1), alpha = 0.01, lr = 0.001):
            
            input1 = Input(shape = input_shape)
            block = CNNBlock(alpha = alpha, name = 'l')
            output1 = block(input1)
            dense0 = Dense(256, activation = 'relu', name = 'd1', kernel_regularizer = l2(alpha))(output1)
            dense1 = Dense(128, activation = 'relu', name = 'extract', kernel_regularizer = l2(alpha))(dense0)
            drop2 = Dropout(0.2)(dense1)
            dense2 = Dense(2, activation = 'softmax')(drop2)
            opt = Adam(lr = lr)
            cnn_model = Model([input1], dense2)
            cnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        
            return cnn_model
            
        pet_model = mymodel(train_images.shape[1:5], 0.01, 0.0005)
        #don't use images whose outcome/score are unavailable
        pet_train = pet_model.fit([train_images[0:300]], seq_diags[0:300], batch_size = 40, epochs = 50,
         shuffle = True, validation_data = ([test_images], seq_test_diags), verbose = 2)
     
        #save extracted features as dict with graph_id's as keys
        bp_train_ex = dict()
        bp_test_ex = dict()
        lt_train_ex = dict()
        lt_test_ex = dict()
        la_train_ex = dict()
        la_test_ex = dict()
        ra_train_ex = dict()
        ra_test_ex = dict()
        rt_train_ex = dict()
        rt_test_ex = dict()
             
        train_ex = dict()
        test_ex = dict()
        
        #specify the layer from which the output is from
        bp_extract_layer = Model(inputs = bp_model.inputs,
                outputs = bp_model.get_layer('extract').output)
        lt_extract_layer = Model(inputs = lt_model.inputs,
                outputs = lt_model.get_layer('extract').output)
        la_extract_layer = Model(inputs = la_model.inputs,
                outputs = la_model.get_layer('extract').output)
        ra_extract_layer = Model(inputs = ra_model.inputs,
                outputs = ra_model.get_layer('extract').output)
        rt_extract_layer = Model(inputs = rt_model.inputs,
                outputs = rt_model.get_layer('extract').output)
        
        train_diags_y = []
        test_diags_y = []
        
        train_scores_y = []
        test_scores_y = []
        
        for gid in train_index:   

            train_diags_y.append(bp_diags_y[gid])
            train_scores_y.append(bp_scores_y[gid])
            bp_ex = bp_extract_layer(bp_images[gid])
            lt_ex = lt_extract_layer(lt_images[gid])
            la_ex = la_extract_layer(la_images[gid])
            ra_ex = ra_extract_layer(ra_images[gid])
            rt_ex = rt_extract_layer(rt_images[gid])
            
            cur_scores = bp_scores[gid]
            cur_diag = bp_diags[gid][:,0].reshape((-1,1))
            bp_train_ex[gid] = bp_ex
            lt_train_ex[gid] = lt_ex
            la_train_ex[gid] = la_ex
            ra_train_ex[gid] = ra_ex
            rt_train_ex[gid] = rt_ex
            train_ex[gid] = np.concatenate([bp_ex, la_ex, lt_ex, ra_ex, rt_ex, cur_scores, cur_diag], axis = 1)
            
        for gid in test_index:   

            test_diags_y.append(bp_diags_y[gid])
            test_scores_y.append(bp_scores_y[gid])
            bp_ex = bp_extract_layer(bp_images[gid])
            lt_ex = lt_extract_layer(lt_images[gid])
            la_ex = la_extract_layer(la_images[gid])
            ra_ex = ra_extract_layer(ra_images[gid])
            rt_ex = rt_extract_layer(rt_images[gid])
            
            cur_scores = bp_scores[gid]
            cur_diag = bp_diags[gid][:,0].reshape((-1,1))
            bp_test_ex[gid] = bp_ex
            lt_test_ex[gid] = lt_ex
            la_test_ex[gid] = la_ex
            ra_test_ex[gid] = ra_ex
            rt_test_ex[gid] = rt_ex
            test_ex[gid] = np.concatenate([bp_ex, la_ex, lt_ex, ra_ex, rt_ex, cur_scores, cur_diag], axis = 1)
            
        #np array
        train_diags_y = np.array(train_diags_y)
        test_diags_y = np.array(test_diags_y)
        train_scores_y = np.array(train_scores_y)
        test_scores_y = np.array(test_scores_y)
        
        #save as dataframe from dictionary
        #define column names
        col_names = []
        for num_feats in range(128*5):
            col_names.append(num_feats+1)
        
        col_names = col_names + ['ADAS', 'MMSE', 'CDR', 'DXCURREN']
        df_train = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in train_ex.items()
                                     for i in range(len(vs))}, orient = 'index', columns=col_names)
        
        df_test = pd.DataFrame.from_dict({(k,i):vs[i] for k,vs in test_ex.items()
                                     for i in range(len(vs))}, orient = 'index', columns=col_names)
            
        
        df_train_gid = [index[0] for index in df_train.index]
        df_test_gid = [index[0] for index in df_test.index]
        
        df_train['graph_id'] = df_train_gid
        df_test['graph_id'] = df_test_gid
        
        #the saved extracted features (node features) include extracted LT and BC,
        #as well as cognitive score (mmse, adas, cdr) and cognitive status
        #and graph_id (so we know which subjects are used as training/test sets)
        train_path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/train_nodes.csv'])
        test_path = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/test_nodes.csv'])
        df_train.to_csv(train_path, index = False)
        df_test.to_csv(test_path, index = False)
        

