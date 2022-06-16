from __future__ import division
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8) # for reproduce

   
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
#from numpy.polynomial.polynomial import polyfit
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib
from IPython.display import SVG, display
import seaborn as sns; sns.set(color_codes=True)



def predict(model, dataset):
    model.eval()
    y_pred_list = {}
    predList = np.arange(0,dataset.shape[0])
    batch_list = []
    
    for i in range(0, dataset.shape[0], batch_size):
        batch = predList[i:i+batch_size]
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
            
            if len(y_pred) == 0:
                continue
            y_pred_adjust = F.softmax(y_pred,dim=-1).data.cpu().numpy()[:,1]

            try:
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_pred_list[i] = []
                y_pred_list[i].extend(y_pred_adjust)
        
    
    return y_pred_list

if __name__ == "__main__":    

    tasks = ['acute_toxic']
    test_df = "./data/test_500_2581.csv"
    per_task_output_units_num = 2
    batch_size = 100
    n_model=10
    
    filename=test_df.replace('.csv','')
    smiles_tasks_df = pd.read_csv(test_df)
    smilesList = smiles_tasks_df.smiles.values
    print("number of all smiles: ",len(smilesList))
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
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    test_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]

    test_tasks_df['cano_smiles'] =canonical_smiles_list

    feature_dicts = save_smiles_dicts(canonical_smiles_list,filename)
    df_result=pd.DataFrame(columns=['smiles'])
    df_result['smiles']=test_tasks_df['smiles']
    df_result['total']=0
    i=0
    for i in range(10):
        best_model = torch.load('saved_models_copy/best_model'+str(i))  
        y_pred_list= predict(best_model, test_tasks_df)   
        df_result[i]=y_pred_list[0]
        df_result['total']=df_result['total']+df_result[i]
        
  
    df_result['predict_average']=df_result['total'].map(lambda x: x/n_model)
    df_result['true']=test_tasks_df['500_toxic']
    
    df_result[['smiles','true','predict_average']].to_csv('./data/predict_output_average.csv')
    