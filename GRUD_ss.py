# -*- coding: utf-8 -*-


### imports correctly running ===================
import torch
# import tensorflow as tf
import torch

# from matplotlib import pyplot
from pypots.optim import Adam
from pypots.classification import GRUD
import pickle as pkl
import numpy as np
# np.random.seed(1)
# from pypots.data import mcar, masked_fill
import pandas as pd
# from pypots.utils.metrics import cal_binary_classification_metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef
# import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")
#====================================================METHOD==========================================
# 8 location features + QIDbaseline + 4thweek QIDS (12th week prediction using 4 weeks location features)
# GRUD input: 8 location features (4 timesteps) + QIDS (Qbaseline+ 4th week QIDS)
# since QIDS we just have data of 2 timesteps we add additional 2 dummy timesteps with 0's (QID0+QID1+QID2+QID3)


#=======================================================================================================

# dfq = pd.read_csv('vectors_Qv_vlen4_with_missingdata_predict_at_12week.csvsync.csv')
# for i in range(len(dfq.columns)):
#     if dfq.columns[i] == 'Qbaseline':
#         dfq.columns.values[i]='QID0'
#     elif dfq.columns[i] == 'Q3':
#         dfq.columns.values[i]='QID1'
#     # elif dfq.columns[i] == 'Q1':
#     #     dfq.columns.values[i]='QID2'
#     dfq['QID2']='nan'
#     dfq['QID3']='nan'
       
   
# dfq.to_csv('vectors_Qv_vlen4_with_missingdata_predict_at_12week.csvsync_upd.csv', index=False)



# rnn_sizes = [8,16]
# batch_sizes = [8, 12, 16, 24, 32]
# lrs = [0.1, 0.001, 0.0001, 0.00001]
# patiences= [5, 10]
# epochs = [100, 200]



rnn_sizes = [32]
batch_sizes = [32]
lrs = [0.0001] 
patiences= [5]
epochs = [200]



vlen=4

featurelist = ['medadherence','safety','tolerance']
dataf = [f'{feature}{i}' for i in range(vlen) for feature in featurelist]
timesteps=vlen
featurenum=len(featurelist)
label=['QIDSlabel']



#======================================================================================




# df = pd.DataFrame(columns=['rnn','bs','lr','pa','epoch','f1','precision','recall','accuracy','specificity','roc-auc'])

for rnn_size in rnn_sizes:
    for batch_size in batch_sizes:
        for lr in lrs:
            for patience in patiences:
                np.random.seed(42)
                torch.manual_seed(42)
                for epoch in epochs:

                    # torch.manual_seed(1)
                    # tf.keras.utils.set_random_seed(1)
                    link = 'vectors_Qv_vlen4_with_missingdata_predict_at_12week.csvsync_upd.csv'
                    data = pd.read_csv(link)
                    # lamba = link[len(link)-5]
                    # print(lamba)
                    # #lamba = link[len(link)-9]
                    # lamba = int(lamba)
                    # lamba = lamba
                   
                    #=========== upsampling ===================
                    # Separate majority and minority classes
                    majority_class = data[data['QIDSlabel'] == 'not improved']
                    minority_class = data[data['QIDSlabel'] == 'improved']
                   
                    # Upsample minority class
                    minority_upsampled = resample(minority_class,
                                                  replace=True,       # Sample with replacement
                                                  n_samples=len(majority_class), # Match number of majority class
                                                  random_state=42)    # For reproducibility
                   
                    # Combine majority class with upsampled minority class
                    upsampled_train_data = pd.concat([majority_class, minority_upsampled])
                   
                    # Shuffle the data to mix the classes well
                    upsampled_train_data = upsampled_train_data.sample(frac=1, random_state=42).reset_index(drop=True)
                    dataup = upsampled_train_data
                    #=========================================
         
                   
                    user_col = 'userid'

                    result_set1 = []
                    result_set2 = []
                   
                    gt_user=[]
                    pred_user=[]
                   
                    all_feature_importances = []

                    for user in data[user_col].unique():
                        print("User=====",user)
                        train_data = data[data[user_col] != user]                        
                        test_data = data[data[user_col] == user]
                       
                        # X_train = train_data[['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']]
                        # y_train = train_data[['manuallabel']]
                        X_train = train_data[dataf]  
                        y_train = train_data[['QIDSlabel']]
                       
                       
                        # X_test = test_data[['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']]
                        # y_test = test_data[['manuallabel']]
                        X_test = test_data[dataf]
                        y_test = test_data[['QIDSlabel']]
                       
                        y_train[y_train == 'improved'] = 1
                        y_test[y_test == 'improved'] = 1                        
                        y_train[y_train == 'not improved'] = 0
                        y_test[y_test == 'not improved'] = 0
                       
                        X_train = np.nan_to_num(X_train)
                        X_test = np.nan_to_num(X_test)
                       
                        # X_train = X_train.reshape((X_train.shape[0], 1, 7))   # soumya : should it be (7, 1) as per GRUD() parameters?
                        # X_test = X_test.reshape((X_test.shape[0], 1, 7))      #
                        X_train_reshape = X_train.reshape((X_train.shape[0], timesteps, featurenum))  # (batch_size,time_steps,number of fetaures) # 85,1,2
                        X_test_reshape = X_test.reshape((X_test.shape[0], timesteps, featurenum))     # (batch_size,time_steps,number of fetaures) # 1,1,2
                       
                       
                        X_train=X_train_reshape
                        X_test=X_test_reshape
                       
                        ## getting training and testing adta together
                        X = np.concatenate((X_train, X_test))  
                        y = np.concatenate((y_train, y_test))
                        # print(X.shape, y.shape)
                        # print('Running test cases for GRUD...')
                       
                        grud = GRUD(X_train.shape[1], # Number of features per time step (see how many fetaures u use for training in your case)
                                    X_train.shape[2],  # Number of time steps (1 in your case)
                                    n_classes=2,   # Assuming binary classification
                                    rnn_hidden_size=rnn_size,#  number of hidden units or neurons in the GRU (Gated Recurrent Unit) layer.
                                    batch_size=batch_size, # number of samples processed before the model's internal parameters are updated.
                                    epochs=epoch, # number of complete passes through the training dataset
                                    optimizer=Adam(lr=lr),  # weight adjustment
                                    patience=patience,   # early stopping ( will stop training if there is no improvement in model performance for consequtive patience number of epochs)
                                    device="cpu")
                       
                        X_train = X_train.astype(float)
                        y_train = y_train.astype(int)
                        y_train = y_train.to_numpy().reshape(-1, 1)   # converts dataframe to numpy array and keeps the shpae same
                        y_test = y_test.to_numpy().reshape(-1, 1)      
                        val_X = []
                        val_y = []
                        val_X = X_test
                        val_y = y_test
                       
                        X_train = X_train.astype(float)
                        y_train = y_train.astype(int)
                        val_X = val_X.astype(float)
                        val_y = val_y.astype(int)
                        y_train= y_train.flatten()
                        val_y= val_y.flatten()
                       
                        val_X = torch.tensor(val_X)
                        val_y = torch.tensor(val_y)
               
                        y_test = y_test.astype(int)
                        y_test= y_test.flatten()
                       
                        ## training dataset
                        dataset_for_training = {
                            "X": torch.tensor(X_train, dtype=torch.float32),
                            "y": torch.tensor(y_train, dtype=torch.float32),
                            }
                        ## validation dataset
                        dataset_for_validating = {
                            "X": torch.tensor(val_X, dtype=torch.float32),
                            "y": torch.tensor(val_y, dtype=torch.float32),
                            }
                        ##  test dataset
                        dataset_for_testing = {
                            "X": torch.tensor(X_test, dtype=torch.float32),
                            "y": torch.tensor(y_test, dtype=torch.float32),
                            }
                       
                       

                        ## model training using train data and validation data
                        history = grud.fit(train_set=dataset_for_training, val_set=dataset_for_validating)
                        
                        ##=========================================================
                        
                        predictions = grud.classify(dataset_for_testing)  # prediction on unseen test data
                        pre_y = []
                    
                        for lst in predictions:
                            lst_ = lst.tolist()
                            t = max(lst_)
                            a = lst_.index(t)
                            pre_y.append(a)   # predicted 

                        y_test = y_test.flatten().tolist()   # actual
                        
                        result_set1.append(y_test)
                        result_set2.append(pre_y)
                        
                        gt_user.append(user)
                        pred_user.append(user)
                        
                    
                    print('LOUO done for all users******************')                   
                    result_set1 = [item for sublist in result_set1 for item in sublist]
                    result_set2 = [item for sublist in result_set2 for item in sublist]
                    mean_f2 = f1_score(result_set1, result_set2)
                    
                    df_result = pd.DataFrame({
                                'user_gt': gt_user,
                                'groundtruth': result_set1,
                                'user_pred': pred_user,
                                'predicted': result_set2
                            })
                    

                    #==================================================
                
             
                    df = pd.read_csv('GRUD_result.csv')
                    #df = pd.DataFrame()
                    df_len = len(df)
 
                    df.loc[df_len, 'rnn']=rnn_size
                    df.loc[df_len, 'bs']=batch_size
                    df.loc[df_len, 'lr']=lr
                    df.loc[df_len, 'pa']=patience
                    df.loc[df_len, 'epoch']=epoch
                    df.loc[df_len, 'f1']=mean_f2
                    df.loc[df_len, 'precision']=precision_score(result_set1, result_set2)
                    df.loc[df_len, 'recall']=recall_score(result_set1, result_set2)
                    df.loc[df_len, 'accuracy']=accuracy_score(result_set1, result_set2)
                    tn, fp, fn, tp = confusion_matrix(result_set1, result_set2).ravel()
                    df.loc[df_len, 'TP']=tp
                    df.loc[df_len, 'FN']=fn
                    df.loc[df_len, 'TN']=tn
                    df.loc[df_len, 'FP']=fp
                    print("TP ={} TN={} FP={} FN={}".format(tp,tn,fp,fn))
                    df.loc[df_len, 'Sum']= tp+tn+fp+fn
                    df.loc[df_len, 'Improved']= tp+fn
                    df.loc[df_len, 'notImproved']= tn+fp
                    specificity = tn / (tn+fp)
                    df.loc[df_len, 'specificity']=specificity
                    df.loc[df_len, 'roc_auc']=roc_auc_score(result_set1, result_set2)
                
                    df.to_csv("GRUD_result.csv", index=False)                    
                    df_result.to_csv("groundtruth_predicted.csv", index=False)
