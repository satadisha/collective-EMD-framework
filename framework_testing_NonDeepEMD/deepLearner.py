import pandas as pd
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import math
import shutil

seed_number = 42
np.random.seed(seed_number)

torch.autograd.set_detect_anomaly(True)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# import numpy as np
# np.random.seed(seed_number)



# 2 output_classes: 'entity'/'non-entity'; so sigmoid transformation would suffice

class NN(nn.Module):
    def __init__(self,input_size):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(input_size,50)
        self.linear2 = nn.Linear(50,25)
        self.linear3 = nn.Linear(25,1)
        # self.linear4 = nn.Linear(6,1)
        self.sigmoid_layer = nn.Sigmoid()
      
    def forward(self, x): 
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))
        x = self.linear3(x)
        out = self.sigmoid_layer(x)
        return out

class EntityClassifier():

    def __init__(self,training_file, to_train, device):

        # # using embedding + syntax features
        # self.combined_feature_list=['length']#,'cap','substring-cap','s-o-sCap','all-cap','non-cap','non-discriminative']+['cf_'+str(i) for i in range(100)]
        # separately using only semantic features
        #self.combined_feature_list=['length']+['cf_'+str(i) for i in range(100)]
        
        self.combined_feature_list= ['length']+['cap','substring-cap','s-o-sCap','all-cap','non-cap','non-discriminative']

        self.relevant_columns = ['normalized_length',
            'normalized_cap',
            'normalized_substring-cap',
            'normalized_s-o-sCap',
            'normalized_all-cap',
            'normalized_non-cap',
            'normalized_non-discriminative'
            ]#+['normalized_cf_'+str(i) for i in range(100)]

        # create scaler
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        
        #initialize the classifier model
        self.classifier = NN(len(self.relevant_columns)).to(device)
        #Loss and Optimizer
        self.ec_criterion = nn.BCELoss(reduction='mean' )
        self.ec_optimizer = optim.Adam(self.classifier.parameters(), lr = 0.00001, weight_decay=0.0001)
        self.ec_batch_size = 32
        self.ec_num_epochs = 2000
        self.patience = 20


        if(to_train):

            self.train = pd.read_csv(training_file,delimiter=",",sep='\s*,\s*')
            #pre-processing : this completes the global average pooling
            
            max_length=self.train['length'].max()
            self.train['normalized_length']= self.train['length']/max_length
            for column in self.combined_feature_list[1:]:
                self.train['normalized_'+column]=self.train[column]/self.train['cumulative']
            
            #Loading the data
            training_inputs_array = self.train[self.relevant_columns].to_numpy()
            training_targets_array = self.train['class'].astype(float).to_numpy()

            # # fit and transform in one step
            training_inputs_array_standardized = self.scaler.fit_transform(training_inputs_array)

            training_inputs = torch.from_numpy(training_inputs_array_standardized).type(torch.float)
            # training_inputs = torch.from_numpy(training_inputs_array).type(torch.float)
            training_targets = torch.from_numpy(training_targets_array).type(torch.float)

            print('Input Shape: ', training_inputs.shape)
            print('Output Shape: ', training_targets.shape)

            dataset = TensorDataset(training_inputs, training_targets)

            train=int(math.ceil(len(training_inputs_array)*0.8))
            val=len(training_inputs_array)-train

            train_ds, val_ds = random_split(dataset, [train, val])

            self.train_loader = DataLoader(train_ds, self.ec_batch_size, shuffle=True)
            self.val_loader = DataLoader(val_ds, val) #will execute in 1 batch

            #Training the model
            end_epoch = self.fit()

            # #Saving the model
            # self.checkpoint = {
            #             'epoch': end_epoch + 1,
            #             'state_dict': self.classifier.state_dict(),
            #             'optimizer': self.ec_optimizer.state_dict()
            #         }

            checkpoint_dir = ""#"entityClassifier/model_checkpoints"
            self.save_ckp(self.checkpoint, True, checkpoint_dir)

        else:
            
            # define checkpoint saved path
            #ckp_path = "entityClassifier/model_checkpoints/classifier_checkpoint_model100.pt" #100
            ckp_path = "classifier_checkpoint_model6.pt" #6

            # load the saved checkpoint
            self.classifier, self.ec_optimizer, self.start_epoch = self.load_ckp(ckp_path, self.classifier, self.ec_optimizer)

    def load_ckp(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    def save_ckp(self,state, is_best, checkpoint_dir):
        f_path = checkpoint_dir + 'classifier_checkpoint_model6.pt' #6
        torch.save(state, f_path)

    def fit(self):
        # Train Network
        history_validation = []
        history_training= []
        no_improvement_counter=0
        best_loss = np.float('inf')
        best_f1 = np.float('-inf')
        for epoch in range(self.ec_num_epochs):
            training_batch_loss=[]
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.unsqueeze(1).to(device=device)

                # forwards
                out = self.classifier(data)

                # print('checking shapes:')
                # print(out.shape)
                # print(targets.shape)

                loss = self.ec_criterion(out, targets)
                training_batch_loss.append(loss.item())
                # print(loss.item())

                # backward
                self.ec_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                # gradient descent or adam step
                self.ec_optimizer.step()
            combined_training_loss = np.mean(training_batch_loss)
            history_training.append(combined_training_loss)

            #Validation: DO NOT BACKPROPAGATE HERE
            validation_batch_loss = []
            labels = []
            prediction = []
            with torch.no_grad():
                for batch_idx, (val_data, val_targets) in enumerate(self.val_loader):
                    val_data = val_data.to(device=device)
                    val_targets = val_targets.unsqueeze(1).to(device=device)
                    out = self.classifier(val_data)

                    # print('checking shapes:')
                    # print(out.shape)
                    # print(val_targets.shape)
                    prediction+=out.reshape(-1).tolist()
                    labels+=[int(elem.item()) for elem in val_targets]

                    # loss = F.mse_loss(out, val_targets) round
                    loss = self.ec_criterion(out, val_targets)
                    validation_batch_loss.append(loss.item())
                    # print(validation_batch_loss)
                combined_validation_loss= np.mean(validation_batch_loss)

                class_prediction = [round(elem) for elem in prediction]
                
                #print(labels)
                #print(class_prediction)
                assert len(class_prediction)==len(labels)
                
                tp = len([elem for idx, elem in enumerate(class_prediction) if((labels[idx]==1)&(elem==1))])
                fp = len([elem for idx, elem in enumerate(class_prediction) if((labels[idx]==0)&(elem==1))])
                fn = len([elem for idx, elem in enumerate(labels) if((elem==1)&(class_prediction[idx]==0))])
                
                #print(tp,fp,fn)

                if((tp+fp)==0):
                    precision = 0
                else:
                    precision = tp/(tp+fp)
                
                if((tp+fn)==0):
                    recall = 0
                else:
                    recall = tp/(tp+fn)
                
                if((precision + recall)==0):
                    f1 = 0
                else:
                    f1 = 2*precision*recall/(precision + recall)

                history_validation.append(combined_validation_loss)
                
                
                #print(len(class_prediction),len(labels))
                #if(((epoch+1)%10==0)|(epoch == (self.ec_num_epochs-1))):
#                     print('Epoch',str(epoch+1),':',combined_training_loss,',',combined_validation_loss)
#                     print('precision:',precision,'recall:',recall,'f1:',f1)
#                     print('=========')
                if(combined_validation_loss<best_loss):
                # if(f1>best_f1):
                    best_loss = combined_validation_loss
                    # best_f1 = f1
                    #print('making this the checkpoint to save')
                    #Saving the model
                    self.checkpoint = {
                                'epoch': epoch + 1,
                                'state_dict': self.classifier.state_dict(),
                                'optimizer': self.ec_optimizer.state_dict()
                            }
                    no_improvement_counter=0
                else:
                    no_improvement_counter+=1
                    if(no_improvement_counter>self.patience):
                        break

        return epoch

    def run(self,candidateBase):

        candidateBase['probability']=-1
        max_length=candidateBase['length'].max()
        candidateBase['normalized_length']= candidateBase['length']/max_length
        for column in self.combined_feature_list[1:]:
            candidateBase['normalized_'+column]=candidateBase[column]/candidateBase['cumulative']

        test_inputs_array = candidateBase[self.relevant_columns].to_numpy()
        test_targets_array = candidateBase['probability'].to_numpy()

        test_inputs_array_standardized = self.scaler.fit_transform(test_inputs_array)

        test_inputs = torch.from_numpy(test_inputs_array_standardized).type(torch.float)
        # test_inputs = torch.from_numpy(test_inputs_array).type(torch.float)
        test_targets = torch.from_numpy(test_targets_array).type(torch.float)

        test_dataset = TensorDataset(test_inputs, test_targets)
        test_loader = DataLoader(test_dataset, len(test_dataset)) #will execute in 1 batch

        #Testing
        prediction=[]
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(device=device)
                # targets = targets.to(device=device)
                out = self.classifier(data)
                print(out.shape)
                prediction=out.reshape(-1)
                print(prediction.shape)

        candidateBase['probability'] = prediction.tolist()
        print(candidateBase['probability'].min(), candidateBase['probability'].max())
        return candidateBase