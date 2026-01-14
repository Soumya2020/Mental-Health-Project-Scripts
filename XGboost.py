#Data Mining
from enum import unique
from math import gamma
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import datetime
import random


import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#usage 05_XGboost.py lambdaval vlen
# Avlens = [int(sys.argv[3])]
# vlens = [int(sys.argv[2])]
# lambdavals = [int(sys.argv[1])]
#from xgBoost_Shweta_Feature_Selection import feature_selection_days_sms_call
vlens=[1]
random.seed(10)
lasttime=0

def get_feature_ranks(original_features):
    original_features_dict=dict(zip(range(len(original_features)),original_features))
    feature_dict=sorted(original_features_dict.items(), key=lambda x: x[1], reverse=True)
    features_rank_tmp=list(zip(*feature_dict))[0]
    features_rank_tmp_dict=dict(zip(range(len(features_rank_tmp)),features_rank_tmp))
    features_rank_tmp_dict=sorted(features_rank_tmp_dict.items(), key=lambda x: x[1])
    features_rank=list(zip(*features_rank_tmp_dict))[0]
    feature_imp_dict=dict(zip(features_rank,original_features))

    # Log the file feature calculation
    logfilename=ori_file_name.replace(".csv",".log")
    file1 = open(logfilename, "a" if os.path.exists(logfilename) else "w")
    L = [str(feature_list)+"\n", str(original_features)+"\n"]
    file1.writelines(L)
    file1.close()

    feature_name_rank_dict=dict(zip(feature_list,features_rank))
    feature_num_dict=dict(zip(features_rank,range(len(feature_list))))
    feature_name_dict=dict(zip(features_rank,feature_list))

    # Filter the selected features.
    feature_name_ordered=[feature_name_dict[x] for x in range(len(feature_list))]
    feature_rank_ordered=[feature_num_dict[x] for x in range(len(feature_list))]
    feature_imp_ordered=[feature_imp_dict[x] for x in range(len(feature_list))]
    return feature_name_rank_dict, feature_name_ordered, feature_rank_ordered, feature_imp_ordered
#print("05_XGboost.py {} {} {}".format(lambdavals,vlens,Avlens))
#for MaxVlen in Avlens:
for VLEN in vlens:
        #if VLEN > MaxVlen: continue
        maxf1=0

        #for ori_file_name in ['result/depmon_vectors_sleep_{}_lambda_sleepBL_vlen3_manual'.format(lambdaval) for lambdaval in [2,3]]:
        # if VLEN == MaxVlen:
        #     filelist=['vectors_Qv_vlen{}_updated_sleeptrendnomissing{}.csv'.format(VLEN,x) for x in lambdavals]
        # else:
        #     filelist=['vectors_Qv_vlen{}_alignedtovlen{}_sleeptrendnomissing{}.csv'.format(VLEN,MaxVlen,x) for x in lambdavals]
        
        filelist=['vectors_Qv_vlen{}_updated_location_variance.csv'.format(VLEN)]
        
        for ori_file_name in filelist:
            print(ori_file_name)
            #data_original = pd.read_csv(ori_file_name+'.csv')
            data_original = pd.read_csv(ori_file_name)
            
            totalsamples=data_original

    #        totalsamples=data_original[data_original['select']]
            dfmetric=pd.DataFrame()
            dfmetric['totalsamples']=[len(totalsamples)]
            numimprovedsamples=len(totalsamples[totalsamples['manuallabel']=='improved'])
            numnonImprovedsamples=len(totalsamples[totalsamples['manuallabel']=='nonImproved'])
            dfmetric['improvedsamples']=[numimprovedsamples]
            dfmetric['nonImprovedsamples']=[numnonImprovedsamples]
            dfmetric['totalusers']=[len(totalsamples['userid'].unique())]
            dfmetric['improvedusers']=[len(totalsamples[totalsamples['manuallabel']=='improved']['userid'].unique())]
            dfmetric['nonImprovedusers']=[len(totalsamples[totalsamples['manuallabel']=='nonImproved']['userid'].unique())]
            if not os.path.exists(ori_file_name+'metrics.csv'):
                dfmetric.to_csv(ori_file_name+'metrics.csv')
            max_value=data_original['Qbaseline'].max()
            
            #df=pd.DataFrame(columns=['sleep','select','log2C','log2g','f1','TP','FN','FP','TN','accuracy','precision','recall'])
            df=pd.DataFrame(columns=['sleep','select','MDepth','Nestimators','min_child_val','gamma_val','subsample_val','colsample_val','f1','TP','FN','FP','TN','accuracy','precision','recall','top_n','feature_rank','feature_names','feature_importance','tmpcols'])

            print(ori_file_name)

            #for sleep in ['None']:
            dfsummary=pd.DataFrame()
            #for sleepf in ['location_variance','time_spent_moving','total_distance','AMS','unique_locations','entropy','normalized_entropy','time_home']: # change 'None' to 'QIDS' as 'None' is used as a setting here
            #for sleepf in ['AMS']:
            #for sleep in ['FA','FB','FBD','SA','DSA','None']:   # None,SA,FA,FB working  # issues with FBD &  DSA
            for sleep in ['FB']:                
                
                print(sleep)
            # Select - use the automatically selected vectors QIDS and corrosponding sleep (Sleep is not selected manually. Sleep is only selected for given QIDS period)
            # Select - use the manually selected vectors - QIDS and corrosponding sleep
                for select in ['select']:
                            #select the data
                    #(data_original[select])&
                    data_original=data_original[(data_original['location_variancebaseline'].notna())]
                    if numnonImprovedsamples>numimprovedsamples:
                        upsample=numnonImprovedsamples/numimprovedsamples
                        upsample=int(upsample+0.5)
                        data_original_improved=data_original[data_original['manuallabel']=='improved']
                        data_original_nonImproved=data_original[data_original['manuallabel']=='nonImproved']
                        data_upsampled_improved=data_original[data_original['manuallabel']=='improved'].sample(frac=upsample, replace=True, random_state=1)
                        data_upsampled_nonImproved=data_original[data_original['manuallabel']=='nonImproved']
                        data_original2=data_original_improved.append(data_original_nonImproved, ignore_index=True)
                        data_upsampled=data_upsampled_improved.append(data_upsampled_nonImproved,ignore_index=True)
                    else:
                        upsample=numimprovedsamples/numnonImprovedsamples
                        upsample=int(upsample+0.5)
                        data_original_improved=data_original[data_original['manuallabel']=='improved']
                        data_upsampled_improved=data_original[data_original['manuallabel']=='improved']
                        data_original_nonImproved=data_original[data_original['manuallabel']=='nonImproved']
                        data_upsampled_nonImproved=data_original[data_original['manuallabel']=='nonImproved'].sample(frac=upsample, replace=True, random_state=1)
                        data_original2=data_upsampled_improved.append(data_original_nonImproved,ignore_index=True)
                        data_upsampled=data_upsampled_improved.append(data_upsampled_nonImproved,ignore_index=True)
                    

                    data_original2['manuallabelBool']=data_original2['manuallabel']=='improved'
                    

                    id_list= data_original2['userid'].unique().tolist()


                    Accuracy_list = []
                    Precision_list = []
                    Recall_list = []

                    timenow=time()*1000
                    looptime=timenow-lasttime
                    lasttime=timenow

                    print(looptime)
                    
                    timenow=time()*1000
                    looptime=timenow-lasttime
                    lasttime=timenow
                    print(looptime)
                    sleepf = ['location_variance','time_spent_moving','total_distance','AMS','unique_locations','entropy','normalized_entropy','time_home','Circadian movement']
                    Q_columns=['Q'+str(x) for x in range(VLEN)]
                    DeltaQ_columns=['DeltaQ'+str(x) for x in range(VLEN)]

                    #FeatureBaselineQ_columns=[sleepf+ 'baseline']
                    FeatureBaselineQ_columns=[ x + 'baseline' for x in (sleepf)]
                    #Featurecolumns=[ x  for x in (sleepf)]
                    Featurecolumns=['location_variance' + str(x) for x in range(VLEN)]+['time_spent_moving' + str(x) for x in range(VLEN)]+['total_distance' + str(x) for x in range(VLEN)]+['AMS' + str(x) for x in range(VLEN)]+[ 'unique_locations' + str(x) for x in range(VLEN)]+['entropy' + str(x) for x in range(VLEN)]+['normalized_entropy' + str(x) for x in range(VLEN)]+['time_home' + str(x) for x in range(VLEN)]+['Circadian movement' + str(x) for x in range(VLEN)]
                    #FeatureDeltaQ_columns=['D'+ sleepf+ str(x) for x in range(VLEN)]
                    FeatureDeltaQ_columns=['D'+ 'location_variance' + str(x) for x in range(VLEN)]+['D'+ 'time_spent_moving' + str(x) for x in range(VLEN)]+['D'+ 'total_distance' + str(x) for x in range(VLEN)]+['D'+ 'AMS' + str(x) for x in range(VLEN)]+['D'+ 'unique_locations' + str(x) for x in range(VLEN)]+['D'+ 'entropy' + str(x) for x in range(VLEN)]+['D'+ 'normalized_entropy' + str(x) for x in range(VLEN)]+['D'+ 'time_home' + str(x) for x in range(VLEN)]+['D'+ 'Circadian movement' + str(x) for x in range(VLEN)]
                    ## fetaure settings #######################################
                    # None = QBaseline + QDelta  
                    # SA = QBaseline + QDelta + Feature
                    # DSA = QBaseline + QDelta + Feature only + FeatureBaseline + FeatureDelta
                    # FA= feature only
                    # FB=Feature only + Featurebaseline
                    # FBD= FeatureBaseline + FeatureDelta + Feature only  # 27
                    # FBD= FeatureBaseline + FeatureDelta  # to do 
                    #############################################################
                    if sleep=='None':
                            S_columns=[]                       
                    elif (sleep=='FA') :
                            #S_columns=[sleepf+str(x) for x in range(VLEN)]
                            S_columns= Featurecolumns
                    elif (sleep=='FB'):
                            #S_columns=[sleepf+str(x) for x in range(VLEN)] + FeatureBaselineQ_columns
                            S_columns= Featurecolumns + FeatureBaselineQ_columns
                    elif (sleep=='FBD'):
                            #S_columns=[sleepf+str(x) for x in range(VLEN)] + FeatureBaselineQ_columns + FeatureDeltaQ_columns
                            S_columns= Featurecolumns + FeatureBaselineQ_columns+FeatureDeltaQ_columns
                    elif (sleep=='SA'):
                            #S_columns=[sleepf+str(x) for x in range(VLEN)]
                            S_columns= Featurecolumns
                    elif (sleep=='DSA'):
                            #S_columns=[sleepf+str(x) for x in range(VLEN)] + FeatureBaselineQ_columns + FeatureDeltaQ_columns
                            S_columns= Featurecolumns + FeatureBaselineQ_columns+FeatureDeltaQ_columns
                    #Create the DeltaQ values
                    for Q_column in Q_columns:
                        data_original2['Delta'+Q_column]=data_original2[Q_column]-data_original2['Qbaseline']
                        data_upsampled['Delta'+Q_column]=data_upsampled[Q_column]-data_upsampled['Qbaseline']

                    DeltaQ_columns+=['Qbaseline']
                    if sleep in ['None','SA','SV','SAV']:
                        DeltaQ_columns=[x.replace('DeltaQ','Q') for x in DeltaQ_columns]
                    # num_features=4
                    # dayval=1
                    feature_list=DeltaQ_columns+S_columns
                    ### SS added for 'None', 'SA','FA','FB'  #############
                    if sleep in ['None','SA','SV','SAV','DSA']:
                        feature_list=DeltaQ_columns+S_columns
                    if sleep in ['FA']:                            
                        feature_list=S_columns
                    if sleep in ['FB']:
                        feature_list=S_columns+FeatureBaselineQ_columns
                    if sleep in ['FBD']:
                        feature_list=S_columns+FeatureBaselineQ_columns+ FeatureDeltaQ_columns  # 45 features
                        #print(feature_list)
                        #print('--------------------------------')
                    #Remove baselines
                    feature_list=[x for x in feature_list if x not in ['SAbaseline','SVbaseline'] ]                    
                    feature_list=sorted(set(feature_list),key=feature_list.index) # Get unique value keeping the order same
                    if len(feature_list) < 1: continue
                    
                    best_n=list(range(2,(1+len(feature_list))))[::-1]
                    max_n=len(feature_list)
                    
                    feature_name_rank_dict={}
                    feature_name_ordered=[]
                    feature_rank_ordered=[]
                    feature_imp_ordered=[]
                    DeltaQ_columns_orig=DeltaQ_columns
                    S_columns_orig=S_columns
                    for fr in best_n: # 2 
                        if fr <max_n:
                            DeltaQ_columns=[x for x in DeltaQ_columns_orig if x in feature_list and feature_name_rank_dict[x]<(fr)]
                            S_columns=[x for x in S_columns_orig if x in feature_list and feature_name_rank_dict[x]<(fr)]
                        else:
                            DeltaQ_columns=[x for x in DeltaQ_columns_orig if x in feature_list ]
                            S_columns=[x for x in S_columns_orig if x in feature_list]
                        
                        tmpcols=DeltaQ_columns+S_columns
                        for MDepth in [3,10,2]:
                        #for MDepth in [3]:
                            # Tuning Parameters
                            #Grid search for Log2C
                            #for MDepth in [4,6,8]:
                            if True:
                                print(MDepth)
                                #Grid search for Log2G
                                for Nestimators in [100,200,400]:
                                #for Nestimators in [100]:
                                    print(Nestimators)

                                    for min_child_val in [1,5]:
                                    #for min_child_val in [1]:
                                        for gamma_val in [0,5]:
                                        #for gamma_val in [0]:
                                            for subsample_val in [1,5,10]:
                                            #for subsample_val in [1]:
                                                for colsample_val in [1,5,10]:
                                                #for colsample_val in [1]:
                                    
                                                    #Crossvalidation - leave one user out
                                                    for select_id in id_list:
                                                        #Initialize the SVM model
                                                        clf = xgb.XGBRegressor(
                                                            objective ='binary:logistic', # binary:hinge
                                                            #colsample_bytree = 1, # colsample_bytree: random subsample of columns when new tree is created select all
                                                            learning_rate = 0.1, # learning_rate = 0.3 : reduce to 0.1 as per shweta's model
                                                            max_depth = MDepth, #
                                                            alpha = 0.005, # Changed from 0 to 0.005 as per shweta ; L1 regularization term on weights. Increasing this value will make model more conservative.
                                                            n_estimators = Nestimators,
                                                            #added as per shweta's code 
                                                            min_child_weight=min_child_val,
                                                            gamma=gamma_val/10,
                                                            subsample=subsample_val/10,
                                                            colsample_bytree=colsample_val/10,
                                                            #reg_alpha=0.005, 
                                                            #nthread=4, #default - use maximum
                                                            scale_pos_weight=1, # default 1
                                                            seed=10
                                                            )
                                                        
                                                        
                                                            #             model= XGBClassifier(
                                                            # learning_rate=0.01,
                                                            # n_estimators=1000,
                                                            # max_depth=max_depth_val,
                                                            # min_child_weight=min_child_val,
                                                            # gamma=gamma_val/10,
                                                            # subsample=subsample_val/10,
                                                            # colsample_bytree=colsample_val/10,
                                                            # reg_alpha=0.005,
                                                            # objective='binary:logistic',
                                                            # nthread=4,
                                                            # scale_pos_weight=1,
                                                            # seed=35)

                                                        #select N-1 users excluding select_id
                                                        data_train=data_upsampled[data_upsampled['userid']!=select_id]
                                                        #data_train=data_original2[data_original2['userid']!=select_id]
                                                        #select Nth user-  select_id
                                                        data_test=data_original2[data_original2['userid']==select_id]

                                                        #Generate Training set
                                                        train_data_X_list=data_train[DeltaQ_columns].to_numpy()/max_value
                                                        if len(S_columns)>0:
                                                            train_data_X_list_sleep=data_train[S_columns].to_numpy()
                                                            scaler = MinMaxScaler()
                                                            # MINMAX normalization
                                                            scaler.fit(train_data_X_list_sleep)
                                                            train_data_X_list_sleep=scaler.transform(train_data_X_list_sleep)
                                                            #train_data_X_list=np.concatenate((train_data_X_list,train_data_X_list_sleep),axis=1)
                                                            if (sleep=='FA') or (sleep=='FB') or (sleep=='FBD'):
                                                                train_data_X_list= train_data_X_list_sleep                                
                                                            elif (sleep=='SA') or (sleep=='DSA'):
                                                                train_data_X_list=np.concatenate((train_data_X_list,train_data_X_list_sleep),axis=1)
                                                        
                                                        train_data_Y_list=data_train['manuallabel'].apply(lambda x: 1.0 if x=='improved' else 0 ).tolist()                   
                                                        
                                                        #Generate Testing set
                                                        test_data_X_list=data_test[DeltaQ_columns].to_numpy()/max_value
                                                        if len(S_columns)>0:
                                                            test_data_X_list_sleep=data_test[S_columns].to_numpy()
                                                            test_data_X_list_sleep=scaler.transform(test_data_X_list_sleep)
                                                            #test_data_X_list=np.concatenate((test_data_X_list,test_data_X_list_sleep),axis=1)
                                                            if (sleep=='FA') or (sleep=='FB') or (sleep=='FBD') :
                                                                test_data_X_list=test_data_X_list_sleep
                                                            elif (sleep=='SA') or (sleep=='DSA'):
                                                                test_data_X_list=np.concatenate((test_data_X_list,test_data_X_list_sleep),axis=1)
                                                        test_data_Y_list=data_test['manuallabelBool'].tolist()
                                                        section_id_list=data_test['sectionid'].tolist()
                                                        
                                                                        
                                                        #print(test_data_Y_list)
                                                        #print(train_data_X_list)
                                                        #print(train_data_Y_list)
                                    
                                                        clf.fit(train_data_X_list, train_data_Y_list)
                                                        y_pred = clf.predict(test_data_X_list)
                                                        y_pred=[x > 0.5 for x in y_pred]
                                                        #print(y_pred)
                                                        count = 0
                                                        for count,section_id in enumerate(section_id_list):

                                                            timenow=time()*1000
                                                            looptime=timenow-lasttime
                                                            lasttime=timenow

                                                           # print(looptime)
                                                            #print('count : ' +str(count) + ' section_id : ' + str(section_id) + ' section_id_list_len : '+ str(len(section_id_list)))
                                                            #print(section_id_list)
                                                            #check = data_original.userid
                                                            #print(select_id)
                                                            #print(check)
                                                            #data_original.loc[(data['userid'] == select_id)&(data['sectionid'] == section_id)]['svm_result'] = y_pred[count]
                                                            data_original2.loc[(data_original2['userid'] == int(select_id)) & (data_original2['sectionid'] == int(section_id)),'ML_result'] = y_pred[count]
                                                            
                                                            if(y_pred[count] == 1.0):
                                                                data_original2.loc[(data_original2['userid'] == int(select_id)) & (data_original2['sectionid'] == int(section_id)),'ML_result_state'] = 'improved'
                                                            else:
                                                                data_original2.loc[(data_original2['userid'] == int(select_id)) & (data_original2['sectionid'] == int(section_id)),'ML_result_state'] = 'nonImproved'
                                                            #check = data_original.loc[(data_original['userid'] == int(select_id)) & (data_original['sectionid'] == int(section_id))]['svm_result']
                                                            
                                                            #= y_pred[count]
                                                            #print(check)
                                                            #print(check)
                                                            #count = count + 1
                                                            print('----------')
                                                            
                                                        
                                                        #print("Accuracy:",metrics.accuracy_score(test_data_Y_list, y_pred))
                                                        
                                                        Accuracy_list.append(metrics.accuracy_score(test_data_Y_list, y_pred))
                                                        psc=metrics.precision_score(test_data_Y_list, y_pred)
                                                        #print("Precision:",psc)
                                                        Precision_list.append(metrics.precision_score(test_data_Y_list, y_pred))
                                                        #print("Recall:",metrics.recall_score(test_data_Y_list, y_pred,pos_label="improved"))
                                                        Recall_list.append(metrics.recall_score(test_data_Y_list, y_pred))
                                                
                                                    # plt.plot(Accuracy_list)
                                                    # plt.savefig(ori_file_name+"accuracy_XGB_{}.png".format(select))
                                                    # plt.close()
                                                    #print('break')
                                                    #print(data)
                                                    data_original2.to_csv(ori_file_name+'_with_XGB_result_{}.csv'.format(select),index=False)
                                                    data_upsampled.to_csv(ori_file_name+'_upsampled_train_{}.csv'.format(select),index=False)
                                                    groundTruthlabels=data_original2['manuallabel'].tolist()
                                                    XGBlabels=data_original2['ML_result_state'].tolist()
                                                    
                                                    #print("check1")
                                                    #print(XGBlabels)
                                                    #print(groundTruthlabels)
                                                    #print("check2")
                                                    
                                                    XGBconfusionmatrics=confusion_matrix(groundTruthlabels, XGBlabels, labels=['improved','nonImproved']).ravel()
                                                    XGBf1=f1_score(groundTruthlabels, XGBlabels, average='macro')
                                                    accuracy=metrics.accuracy_score(groundTruthlabels, XGBlabels)
                                                    precision=metrics.precision_score(groundTruthlabels, XGBlabels,pos_label="improved")
                                                    recall=metrics.recall_score(groundTruthlabels, XGBlabels,pos_label="improved")
                                                    #['select','log2C','log2g','f1','TN','FP','FN','PP','accuracy','precision','recall']
                                                    if fr==max_n:
                                                        original_features=clf.feature_importances_
                                                        feature_name_rank_dict, feature_name_ordered, feature_rank_ordered, feature_imp_ordered=get_feature_ranks(original_features)
                                                    df.loc[len(df)]=[sleep,select,MDepth,Nestimators,min_child_val, gamma_val, subsample_val, colsample_val,XGBf1]+list(XGBconfusionmatrics)+[accuracy,precision,recall,fr,feature_rank_ordered[:(fr)],feature_name_ordered[:(fr)],feature_imp_ordered[:(fr)],tmpcols]
                                                    #print('XGB Depmon')
                                                    #print(XGBconfusionmatrics)
                                                    #rint(XGBf1)
                                                df.to_csv(ori_file_name+'XGB.csv',index=False)
                        # Update the summary for setting=sleep and number of features selected
                        dftmp=df[(df['sleep']==sleep)&(df['top_n']==(fr))]
                        ii=dftmp['f1'].idxmax()
                        dftmp2=dftmp.loc[[ii]]
                        dfsummary=dfsummary.append(dftmp2,ignore_index=True)
                        if fr==max_n:
                            feature_name_ordered=dftmp2['feature_names'].tolist()[0]
                            feature_rank_ordered=dftmp2['feature_rank'].tolist()[0]
                            feature_imp_ordered=dftmp2['feature_importance'].tolist()[0]
                            feature_name_rank_dict=dict(zip(feature_name_ordered,feature_rank_ordered))
                    #Save the combined summary file.
                    dfsummary.to_csv(ori_file_name+'XGB_summary.csv',index=False)