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

from sklearn.metrics import mean_squared_error
from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
# weekbefore=1
random.seed(10)
lasttime=0

def get_feature_ranks(original_features):
  #if len(original_features)>1:
    original_features_dict=dict(zip(range(len(original_features)),original_features))       # importances for every feature
    feature_dict=sorted(original_features_dict.items(), key=lambda x: x[1], reverse=True)   # sort importances[index, importance]
    features_rank_tmp=list(zip(*feature_dict))[0]                                           # index of importances
    features_rank_tmp_dict=dict(zip(range(len(features_rank_tmp)),features_rank_tmp))
    features_rank_tmp_dict=sorted(features_rank_tmp_dict.items(), key=lambda x: x[1])         # sorting ranks
    features_rank=list(zip(*features_rank_tmp_dict))[0]
    feature_imp_dict=dict(zip(features_rank,original_features))        # importances

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
    #print(len(feature_list))
    #print(feature_name_dict)
    feature_name_ordered=[feature_name_dict[x] for x in range(len(feature_list))]
    feature_rank_ordered=[feature_num_dict[x] for x in range(len(feature_list))]
    feature_imp_ordered=[feature_imp_dict[x] for x in range(len(feature_list))]
    return feature_name_rank_dict, feature_name_ordered, feature_rank_ordered, feature_imp_ordered
    

for VLEN in vlens:
        
        filelist=['final_ios_android_vlen_{}.csv'.format(VLEN)]
        
        for ori_file_name in filelist:
            print(ori_file_name)
            #data_original = pd.read_csv(ori_file_name+'.csv')
            data_original = pd.read_csv(ori_file_name)
            
            totalsamples=data_original

    #        totalsamples=data_original[data_original['select']]
            dfmetric=pd.DataFrame()
            dfmetric['totalsamples']=[len(totalsamples)]
            numimprovedsamples=len(totalsamples[totalsamples['finallabel']=='improved'])
            numnonImprovedsamples=len(totalsamples[totalsamples['finallabel']=='nonImproved'])
            dfmetric['improvedsamples']=[numimprovedsamples]
            dfmetric['nonImprovedsamples']=[numnonImprovedsamples]
            dfmetric['totalusers']=[len(totalsamples['userid'].unique())]
            dfmetric['improvedusers']=[len(totalsamples[totalsamples['finallabel']=='improved']['userid'].unique())]
            dfmetric['nonImprovedusers']=[len(totalsamples[totalsamples['finallabel']=='nonImproved']['userid'].unique())]
            if not os.path.exists(ori_file_name+'metrics.csv'):
                dfmetric.to_csv(ori_file_name+'metrics.csv')
            max_value=data_original['Qbaseline'].max()
            
            #df=pd.DataFrame(columns=['sleep','select','log2C','log2g','f1','TP','FN','FP','TN','accuracy','precision','recall'])
            df=pd.DataFrame(columns=['sleep','select','bootstrap_input','max_depth','max_features','min_samples_leaf','min_samples_split','n_estimators','f1','TN','FP','FN','PP','accuracy','precision','recall','top_n','feature_rank','feature_names','feature_importance','tmpcols'])

            print(ori_file_name)

            #for sleep in ['None']:
            dfsummary=pd.DataFrame()

            for sleep in ['FA','FB','SA','None']:   # None,SA,FA,FB working  # issues with FBD &  DSA
            # for sleep in ['SA']:                
                    print(sleep)
                    for select in ['select']:
                      #select the data
                        #data_original=data_original[(data_original['SAbaseline'].notna())]
                        data_original=data_original[(data_original['location_variancebaseline'].notna())]
                        if numnonImprovedsamples>numimprovedsamples:
                            upsample=numnonImprovedsamples/numimprovedsamples
                            upsample=int(upsample+0.5)
                            data_original_improved=data_original[data_original['finallabel']=='improved']
                            data_original_nonImproved=data_original[data_original['finallabel']=='nonImproved']
                            data_upsampled_improved=data_original[data_original['finallabel']=='improved'].sample(frac=upsample, replace=True, random_state=1)
                            data_upsampled_nonImproved=data_original[data_original['finallabel']=='nonImproved']
                            data_original2=data_original_improved.append(data_original_nonImproved, ignore_index=True)
                            data_upsampled=data_upsampled_improved.append(data_upsampled_nonImproved,ignore_index=True)
                        else:
                            upsample=numimprovedsamples/numnonImprovedsamples
                            upsample=int(upsample+0.5)
                            data_original_improved=data_original[data_original['finallabel']=='improved']
                            data_upsampled_improved=data_original[data_original['finallabel']=='improved']
                            data_original_nonImproved=data_original[data_original['finallabel']=='nonImproved']
                            data_upsampled_nonImproved=data_original[data_original['finallabel']=='nonImproved'].sample(frac=upsample, replace=True, random_state=1)
                            data_original2=data_upsampled_improved.append(data_original_nonImproved,ignore_index=True)
                            data_upsampled=data_upsampled_improved.append(data_upsampled_nonImproved,ignore_index=True)
                        
    
                        data_original2['finallabelBool']=data_original2['finallabel']=='improved'
                        
    
                        id_list= data_original2['userid'].unique().tolist()
    
    
                        Accuracy_list = []
                        Precision_list = []
                        Recall_list = []
    

                        
                        sleepf = ['location_variance','time_spent_moving','total_distance','AMS','unique_locations','entropy','normalized_entropy','time_home']
                        Q_columns=['Q']
                        DeltaQ_columns=['DeltaQ'+str(x) for x in range(VLEN)]
                        Featurecolumns=[ x  for x in (sleepf)]

                        FeatureBaselineQ_columns=[ x + 'baseline' for x in (sleepf)]
                        
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
                        elif (sleep=='SA'):
                                #S_columns=[sleepf+str(x) for x in range(VLEN)]
                                S_columns= Featurecolumns
                        #Create the DeltaQ values
                        for Q_column in Q_columns:
                            data_original2['Delta'+Q_column]=data_original2[Q_column]-data_original2['Qbaseline']
                            data_upsampled['Delta'+Q_column]=data_upsampled[Q_column]-data_upsampled['Qbaseline']
    
                        DeltaQ_columns+=['Qbaseline']
                        if sleep in ['None','SA','SV','SAV']:
                            DeltaQ_columns=[x.replace('DeltaQ0','Q') for x in DeltaQ_columns]
                        # num_features=4
                        # dayval=1                        
                        
                        #feature_list=DeltaQ_columns+S_columns
                        
                        ### SS added for 'None', 'SA','FA','FB'  #############
                        if sleep in ['None','SA','SV','SAV','DSA']:
                            feature_list=DeltaQ_columns+S_columns
                        if sleep in ['FA']:                            
                            feature_list=S_columns
                        if sleep in ['FB']:
                            feature_list=S_columns+FeatureBaselineQ_columns

                            
                            
                        #feature_list=S_columns
                        #feature_list=[x for x in feature_list if x not in ['SAbaseline','SVbaseline'] ] 
                        
                        
                        #Remove baselines
                        feature_list=sorted(set(feature_list),key=feature_list.index) # Get unique value keeping the order same
                        print(feature_list)
                        if len(feature_list) < 1: continue
                        #original_features=feature_selection_days_sms_call(feature_list,ori_file_name,VLEN) #unsorted
    
                        best_n=list(range(2,(1+len(feature_list))))[::-1]
                        #if sleep in ['FA']:
                          #best_n=[1]
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
                            for bootstrap_input in [True,False]:
                                print(bootstrap_input)
                                for max_depth_input in [10, 20, None]:
                                # for max_depth_input in [10]:
                                    print(max_depth_input)
                                    for max_features_input in ['auto', 'sqrt']:
                                    # for max_features_input in ['auto']:
                                        print(max_features_input)
                                        for min_samples_leaf_input in [1, 2, 4]:
                                        # for min_samples_leaf_input in [1]:
                                            print(min_samples_leaf_input)
                                            for min_samples_split_input in [2, 5, 10]:
                                            # for min_samples_split_input in [2]:
                                                print(min_samples_split_input)
                                                for n_estimators_input in [100,200, 400]:
                                                # for n_estimators_input in [100]:
                                                    print(n_estimators_input)
                                            
                                                    #Crossvalidation - leave one user out
                                                    for select_id in id_list:
                                                        #Initialize the RF model
                                                        clf=RandomForestClassifier(bootstrap=bootstrap_input,max_depth=max_depth_input,max_features=max_features_input,min_samples_leaf=min_samples_leaf_input,
                                                            min_samples_split=min_samples_split_input,n_estimators=n_estimators_input)

    
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
                                                            scaler.fit(train_data_X_list_sleep)
                                                            train_data_X_list_sleep=scaler.transform(train_data_X_list_sleep)
                                                            #train_data_X_list=np.concatenate((train_data_X_list,train_data_X_list_sleep),axis=1)
                                                            if (sleep=='FA') or (sleep=='FB') or (sleep=='FBD'):
                                                                train_data_X_list= train_data_X_list_sleep                                
                                                            elif (sleep=='SA') or (sleep=='DSA'):
                                                                train_data_X_list=np.concatenate((train_data_X_list,train_data_X_list_sleep),axis=1)
                                                        
                                                        train_data_Y_list=data_train['finallabel']#.apply(lambda x: 1.0 if x=='improved' else 0 ).tolist()                   
                                                        
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
                                                        test_data_Y_list=data_test['finallabel'].tolist()
                                                        section_id_list=data_test['sectionid'].tolist()
     
                                    
                                                        clf.fit(train_data_X_list, train_data_Y_list)
                                                        y_pred = clf.predict(test_data_X_list)
                                                        #print(y_pred)
                                                        count = 0
                                                        for count,section_id in enumerate(section_id_list):
    
                                                            timenow=time()*1000
                                                            looptime=timenow-lasttime
                                                            lasttime=timenow
    
                                                            data_original2.loc[(data_original2['userid'] == int(select_id)) & (data_original2['sectionid'] == int(section_id)),'ML_result'] = y_pred[count]
                                                            
                                                            if(y_pred[count] == 1.0):
                                                                data_original2.loc[(data_original2['userid'] == int(select_id)) & (data_original2['sectionid'] == int(section_id)),'ML_result_state'] = 'improved'
                                                            else:
                                                                data_original2.loc[(data_original2['userid'] == int(select_id)) & (data_original2['sectionid'] == int(section_id)),'ML_result_state'] = 'nonImproved'
                                                            #check = data_original.loc[(data_original['userid'] == int(select_id)) & (data_original['sectionid'] == int(section_id))]['svm_result']
                
                                                        
                                                        Accuracy_list.append(metrics.accuracy_score(test_data_Y_list, y_pred))
                                                        psc=metrics.precision_score(test_data_Y_list, y_pred,pos_label="improved")
                                                        Precision_list.append(metrics.precision_score(test_data_Y_list, y_pred,pos_label="improved"))
                                                        #print("Recall:",metrics.recall_score(test_data_Y_list, y_pred,pos_label="improved"))
                                                        Recall_list.append(metrics.recall_score(test_data_Y_list, y_pred,pos_label="improved"))
                                                
                                                    data_original2.to_csv(ori_file_name+'_with_RF_result_{}.csv'.format(select),index=False)
                                                    data_upsampled.to_csv(ori_file_name+'_upsampled_train_{}.csv'.format(select),index=False)
                                                    groundTruthlabels=data_original2['finallabel'].tolist()
                                                    RFlabels=data_original2['ML_result'].tolist()
                                                    
                                                    #print("check1")
                                                    #print(XGBlabels)
                                                    #print(groundTruthlabels)
                                                    #print("check2")
                                                    
                                                    RFconfusionmatrics=confusion_matrix(groundTruthlabels, RFlabels, labels=['improved','nonImproved']).ravel()
                                                    RFf1=f1_score(groundTruthlabels, RFlabels, average='macro')
                                                    accuracy=metrics.accuracy_score(groundTruthlabels, RFlabels)
                                                    precision=metrics.precision_score(groundTruthlabels, RFlabels,pos_label="improved")
                                                    recall=metrics.recall_score(groundTruthlabels, RFlabels,pos_label="improved")
                                                    #['select','log2C','log2g','f1','TN','FP','FN','PP','accuracy','precision','recall']
                                                    #Save the feature ranks
                                                    if fr==max_n:
                                                        original_features=clf.feature_importances_
                                                        feature_name_rank_dict, feature_name_ordered, feature_rank_ordered, feature_imp_ordered=get_feature_ranks(original_features)
                                                    df.loc[len(df)]=[sleep,select,bootstrap_input,max_depth_input,max_features_input,min_samples_leaf_input,min_samples_split_input,n_estimators_input,RFf1]+list(RFconfusionmatrics)+[accuracy,precision,recall,fr,feature_rank_ordered[:(fr)],feature_name_ordered[:(fr)],feature_imp_ordered[:(fr)],tmpcols]
                                                    #print('RF Depmon')
                                                    #print(RFconfusionmatrics)
                                                    #print(RFf1)
                                                df.to_csv(ori_file_name+'RF.csv',index=False)
                            # Update the summary for setting=sleep and number of features selected
                            dftmp=df[(df['sleep']==sleep)&(df['top_n']==(fr))]
                            ii=dftmp['f1'].idxmax()
                            dftmp2=dftmp.loc[[ii]]
                            dfsummary=dfsummary.append(dftmp2,ignore_index=True)
    
                            #If it is first run select all features and calculate importance for given VLEN, lambda
                            # Then save the features for subsiquent models.
                            if fr==max_n:
                                feature_name_ordered=dftmp2['feature_names'].tolist()[0]
                                feature_rank_ordered=dftmp2['feature_rank'].tolist()[0]
                                feature_imp_ordered=dftmp2['feature_importance'].tolist()[0]
                                feature_name_rank_dict=dict(zip(feature_name_ordered,feature_rank_ordered))
    
                        #Save the combined summary file.
                        dfsummary.to_csv(ori_file_name+'RF_summary.csv',index=False)