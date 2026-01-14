
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pypots.classification import GRUD
from pypots.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef
# from FCNN import SimpleFCNN, train_model # Assuming you have a SimpleFCNN class in FCNN.py
import warnings
warnings.filterwarnings("ignore")



#==========================================================================================
# Define the FCNN model (for non-sequential data)
class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
#========================================================================================


rnn_sizes = [8,16]
batch_sizes = [8, 12, 16, 24, 32]
lrs = [0.1, 0.001, 0.0001, 0.00001]
patiences= [5, 10]
epochs_list = [100, 200]
vlen=4


# rnn_sizes = [32]
# batch_sizes = [32]
# lrs = [0.0001]
# patiences = [2]
# epochs_list = [2]
# vlen=4






# Sequential and Nonsequential Data Columns=========================================
featurelist = ['medadherence','safety','tolerance']
dataf = [f'{feature}{i}' for i in range(vlen) for feature in featurelist]
seq_features = dataf
nonseq_features = ['Qbaseline']
label_col = ['qidslabel']

timesteps=vlen
featurenum=len(featurelist)
num_nonseq_features= len(nonseq_features)






#==========================================================================================
class CombinedModel(nn.Module):
    # def __init__(self, grud, fcnn_hidden_size, combined_hidden_size, grud_hidden_size):
    def __init__(self, grud, fcnn_hidden_size, combined_hidden_size, grud_hidden_size, num_nonseq_features):
        super(CombinedModel, self).__init__()

        self.fcnn = SimpleFCNN(input_size=len(nonseq_features), 
                               hidden_size=fcnn_hidden_size, 
                               output_size=grud_hidden_size)
        
        self.combined_fc = nn.Sequential(
            nn.Linear(1 + fcnn_hidden_size, combined_hidden_size),
            nn.ReLU(),
            nn.Linear(combined_hidden_size, 1)  # Binary classification
        )
    def forward(self, grud_out, nonseq_input):
        # print("grud_out shape={}    noseq_input shape={}\n".format(grud_out.shape,nonseq_input.shape))
        
        fcnn_out = self.fcnn(nonseq_input)
        # print("fcnn_out shape={}\n".format(fcnn_out.shape))
        
        combined_input = torch.cat((grud_out, fcnn_out), dim=1)
        # print("combined_input shape={}\n".format(combined_input.shape))
    
        final_output = self.combined_fc(combined_input)
        # print("final_output shape={}\n".format(final_output.shape))
        return final_output
#=============================================================================================



# Load Dataset
data = pd.read_csv('vectors_Qv_vlen4_with_missingdata_predict_at_12week.csvsync.csv')


# Initialize Metrics Storage
all_results = []


for rnn_size in rnn_sizes:
    for batch_size in batch_sizes:
        for lr in lrs:
            for patience in patiences:
                for epochs in epochs_list:
                    torch.manual_seed(1)
                    # Leave-One-User-Out Cross-Validation
                    user_col = 'userid'
                    
                    result_set1 = []
                    result_set2 = []
                    
                    gt_user=[]
                    pred_user=[]

                    print('LOUO done for all users******************')                     
                    for user in data[user_col].unique():
                        print(f"Processing User: {user}")
                    
                        # Split data
                        train_data = data[data[user_col] != user]
                        test_data = data[data[user_col] == user]
                        
                        X_train = train_data[seq_features]  
                        y_train = train_data[['QIDSlabel']]
                        
                        X_test = test_data[seq_features]
                        y_test = test_data[['QIDSlabel']]
                        
                        X_train = np.nan_to_num(X_train)
                        X_test = np.nan_to_num(X_test)
                    
                        # Sequential Data
                        # X_train_seq = train_data[seq_features].values.reshape(-1, timesteps, len(seq_features))
                        # X_test_seq = test_data[seq_features].values.reshape(-1, timesteps, len(seq_features))
                        X_train_seq = X_train.reshape((X_train.shape[0], timesteps, featurenum))  # (batch_size,time_steps,number of fetaures) # 85,1,2
                        X_test_seq = X_test.reshape((X_test.shape[0], timesteps, featurenum))     # (batch_size,time_steps,number of fetaures) # 1,1,2
                        
                        X_train=X_train_seq
                        X_test=X_test_seq
                    
                        # Nonsequential Data
                        X_train_nonseq = train_data[nonseq_features].values
                        X_test_nonseq = test_data[nonseq_features].values
                    

                        y_train[y_train == 'improved'] = 1
                        y_test[y_test == 'improved'] = 1                        
                        y_train[y_train == 'not improved'] = 0
                        y_test[y_test == 'not improved'] = 0
                        
                        ## getting training and testing adta together
                        X = np.concatenate((X_train, X_test))  
                        y = np.concatenate((y_train, y_test))
                        
                        # Initialize GRUD
                        grud = GRUD(X_train.shape[1], # Number of features per time step (see how many fetaures u use for training in your case)
                                    X_train.shape[2],  # Number of time steps (1 in your case)
                                    n_classes=2,   # Assuming binary classification
                                    rnn_hidden_size=rnn_size,#  number of hidden units or neurons in the GRU (Gated Recurrent Unit) layer.
                                    batch_size=batch_size, # number of samples processed before the model's internal parameters are updated.
                                    epochs=epochs, # number of complete passes through the training dataset
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
                        # Train GRUD
                        grud.fit({"X": X_train_seq, "y": y_train}, {"X": X_test_seq, "y": y_test})

                        # GRUD predictions
                        grud_out_train = torch.tensor(grud.classify({"X": X_train_seq}), dtype=torch.float32)
                        grud_out_test = torch.tensor(grud.classify({"X": X_test_seq}), dtype=torch.float32)
                        
                        grud_out_train = grud_out_train[:, 1].unsqueeze(1)  # Shape becomes [86, 1]
                        grud_out_test = grud_out_test[:, 1].unsqueeze(1)    # Shape becomes [86, 1]

                        
                    
                        # # Convert to Tensors
                        X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
                        X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
                        X_train_nonseq_tensor = torch.tensor(X_train_nonseq, dtype=torch.float32)
                        X_test_nonseq_tensor = torch.tensor(X_test_nonseq, dtype=torch.float32)
                        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).flatten()
                        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).flatten()
                    
                        # Initialize Combined Model
                        model = CombinedModel(grud,
                            fcnn_hidden_size=rnn_size,
                            combined_hidden_size=rnn_size,
                            grud_hidden_size=rnn_size,
                            num_nonseq_features=len(nonseq_features)
                        )
                    
                        # Loss and Optimizer
                        criterion = nn.BCEWithLogitsLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                        # Training Loop
                        for epoch in range(epochs):
                            model.train()
                            
                            # Forward pass                            
                            combined_outputs = model(grud_out_train, X_train_nonseq_tensor).flatten()
                            loss = criterion(combined_outputs, y_train_tensor)
                    
                            # Backward pass and optimization
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
                    
                        # Evaluate combined model
                        model.eval()
                        with torch.no_grad():
                            test_outputs = model(grud_out_test, X_test_nonseq_tensor).flatten()
                            predicted = (torch.sigmoid(test_outputs) > 0.5).float().int().tolist() 

                        y_test = y_test.flatten().tolist()   # actual
                        
                        result_set1.append(y_test)
                        result_set2.append(predicted)
                        
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
                    df = pd.read_csv('GRUD_FCNN_result.csv')
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
                
                    df.to_csv("GRUD_FCNN_result.csv", index=False)                    
                    df_result.to_csv("groundtruth_predicted.csv", index=False)                   
                    
