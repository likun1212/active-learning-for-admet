import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8) # for reproduce
from sklearn.model_selection import KFold,StratifiedKFold
   
import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
#then import my own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight


from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score


# from rdkit.Chem import rdMolDescriptors, MolSurf
# from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
#%matplotlib inline
from numpy.polynomial.polynomial import polyfit
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib
from IPython.display import SVG, display
import seaborn as sns; sns.set(color_codes=True)

        
        
def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    #shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.cano_smiles.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
#         print(torch.Tensor(x_atom).size(),torch.Tensor(x_bonds).size(),torch.cuda.LongTensor(x_atom_index).size(),torch.cuda.LongTensor(x_bond_index).size(),torch.Tensor(x_mask).size())
        
        model.zero_grad()
        # Step 4. Compute your loss function. (Again, Torch wants the target wrapped in a variable)
        loss = 0.0
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
#             validInds = np.where(y_val != -1)[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            
            loss += loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
        # Step 5. Do the backward pass and update the gradient
#             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)
        loss.backward()
        optimizer.step()
def eval(model, dataset):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.loc[test_batch,:]
        smiles_list = batch_df.cano_smiles.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
#             validInds = np.where((y_val=='0') | (y_val=='1'))[0]
#             print(validInds)
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
#             print(validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
            
            y_pred_adjust = F.softmax(y_pred_adjust,dim=-1).data.cpu().numpy()[:,1]
            
            
            
            losses_list.append(loss.cpu().detach().numpy())
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
#             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)            
    test_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(tasks))]
    test_prc = [auc(precision_recall_curve(y_val_list[i], y_pred_list[i])[1],precision_recall_curve(y_val_list[i], y_pred_list[i])[0]) for i in range(len(tasks))]
#     test_prc = auc(recall, precision)
    test_precision = [precision_score(y_val_list[i],
                                     (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    
    
    test_recall = [recall_score(y_val_list[i],
                               (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    test_loss = np.array(losses_list).mean()
    
    return test_roc, test_prc, test_precision, test_recall, test_loss



if __name__ == "__main__":

    tasks = ['acute_toxic']
    raw_filename = "./data/acutetoxic.csv"
    filename = raw_filename.replace('.csv','')
    prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
    
    
    #under_sampling
    # from imblearn.under_sampling import RandomUnderSampler
    # from collections import Counter
    # rus = RandomUnderSampler(random_state=0, replacement=True)
    #smiles_tasks_df_1 = pd.read_csv(raw_filename)
    # X_resampled, y_resampled = rus.fit_resample(np.asarray(smiles_tasks_df_1['smiles']).reshape(-1, 1),smiles_tasks_df_1['acute_toxic'])

    # print(sorted(Counter(y_resampled).items()))
    # sys.stdout.flush()
    # smiles_list=X_resampled[:,0]
    # labels=y_resampled
    # smiles_tasks_df=pd.DataFrame()
    # smiles_tasks_df['smiles']=smiles_list
    # smiles_tasks_df['acute_toxic']=labels
    smiles_tasks_df = pd.read_csv(raw_filename)   #if use under_sampling, need comment this line
    smilesList = smiles_tasks_df.smiles.values
    #print("number of all smiles: ",len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:        
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
           # print("not successfully processed smiles: ", smiles)
           # sys.stdout.flush()        
            pass
    #print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
    # print(smiles_tasks_df)
    smiles_tasks_df['cano_smiles'] =canonical_smiles_list
    assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)

    smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())<101]
    uncovered = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())>100]

    smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)]
    feature_dicts = save_smiles_dicts(smilesList,filename)

    remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    
    
    
    
    #different train and valid
    kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42) 
    training_data=remained_df
    model_number=0
    for train_index , valid_index in kf.split(training_data["cano_smiles"],training_data["acute_toxic"]): 
        train_df=training_data[training_data.index.isin(train_index)]
        valid_df=training_data.drop(train_df.index) 
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        
        # model param preset
        random_seed = 188
        random_seed = int(time.time())
        start_time = str(time.ctime()).replace(':','-').replace(' ','_')
        start = time.time()

        batch_size = 100
        epochs = 800
        p_dropout = 0.1
        fingerprint_dim = 150

        radius = 3
        T = 2
        weight_decay = 2.9 # also known as l2_regularization_lambda
        learning_rate = 3.5
        per_task_output_units_num = 2 # for classification model with 2 classes
        output_units_num = len(tasks) * per_task_output_units_num
            
        weights = []
        for i,task in enumerate(tasks):    
            negative_df = remained_df[remained_df[task] == 0][["smiles",task]]
            positive_df = remained_df[remained_df[task] == 1][["smiles",task]]
            weights.append([(positive_df.shape[0]+negative_df.shape[0])/negative_df.shape[0],\
                            (positive_df.shape[0]+negative_df.shape[0])/positive_df.shape[0]])
            
        
        training_data = remained_df
        
        
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[0]],feature_dicts)
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]

        loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight),reduction='mean') for weight in weights]
        model = Fingerprint(radius, T, num_atom_features,num_bond_features,
                    fingerprint_dim, output_units_num, p_dropout)
        model.cuda()
        # tensorboard = SummaryWriter(log_dir="runs/"+start_time+"_"+prefix_filename+"_"+str(fingerprint_dim)+"_"+str(p_dropout))

        # optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        #print(params)
        sys.stdout.flush()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)
    
        best_param ={}
        best_param["roc_epoch"] = 0
        best_param["loss_epoch"] = 0
        best_param["valid_roc"] = 0
        best_param["valid_loss"] = 9e8

        for epoch in range(epochs):    
            train_roc, train_prc, train_precision, train_recall, train_loss = eval(model, train_df)
            valid_roc, valid_prc, valid_precision, valid_recall, valid_loss = eval(model, valid_df)
            train_roc_mean = np.array(train_roc).mean()
            valid_roc_mean = np.array(valid_roc).mean()
            

            if valid_roc_mean > best_param["valid_roc"]:
                best_param["roc_epoch"] = epoch
                best_param["valid_roc"] = valid_roc_mean
                if valid_roc_mean > 0.75:
                    torch.save(model, 'saved_models_copy/model_'+str(model_number)+'_'+prefix_filename+'_'+str(epoch)+'.pt')     
                      
            if valid_loss < best_param["valid_loss"]:
                best_param["loss_epoch"] = epoch
                best_param["valid_loss"] = valid_loss

            print("EPOCH:\t"+str(epoch)+'\n'\
                +"train_roc"+":"+str(train_roc)+'\n'\
                +"valid_roc"+":"+str(valid_roc)+'\n'\
        #         +"train_roc_mean"+":"+str(train_roc_mean)+'\n'\
        #         +"valid_roc_mean"+":"+str(valid_roc_mean)+'\n'\
                )
            sys.stdout.flush()
            if (epoch - best_param["roc_epoch"] >18) and (epoch - best_param["loss_epoch"] >28):        
                break
                
            train(model, train_df, optimizer, loss_function)
        
        print("best model:", best_param["roc_epoch"], best_param["valid_roc"]) 
          
        sys.stdout.flush()
        try:
            best_model_file='saved_models_copy/model_'+str(model_number)+'_'+prefix_filename+'_'+str(best_param["roc_epoch"])+'.pt'
            dirname, filename=os.path.split(best_model_file)
            new_file=os.path.join(dirname,'best_model'+str(model_number))
            os.rename(best_model_file, new_file)
            model_number=model_number+1
        except:
            pass
    
