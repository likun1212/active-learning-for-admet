# -*- encoding: utf-8 -*-
"""
@File    :   al.py
@Time    :   2022/06/24 15:41:49
@Author  :   likun.yang 
@Contact :   likun_yang@foxmail.com
@Description: Al framework
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score

from AttentiveFP import (
    Fingerprint,
    get_smiles_array,
    get_smiles_dicts,
    moltosvg_highlight,
    save_smiles_dicts,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import time

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type("torch.cuda.FloatTensor")
# from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy

import pandas as pd


class Acquisiton:
    def __init__(self, pool):
        self.pool = pool

    def random(self, sample_size=1000):

        return self.pool.sample(n=sample_size, random_state=42)

    def entropy_sample(self, props, sample_size=500):
        from scipy.stats import entropy

        props = props.reshape((-1, 1))
        props = np.hstack((1 - props, props))
        entropies = entropy(props, axis=1, base=2)
        _pool = self.pool
        _pool["entropy"] = entropies
        _pool.sort_values("entropy", inplace=True)
        return _pool.iloc[-sample_size:]


class Al:
    def __init__(
        self, pool, test_set, iters=8, initial_frac=0.1, batch_frac=0.05, epochs=100
    ) -> None:
        self.pool = pool
        self.test_set = test_set
        self.iters = iters
        self.initial_size = int(initial_frac * len(pool))
        self.batch_size = int(batch_frac * (len(pool) - self.initial_size))
        self.epochs = epochs
        self.train_data = pd.DataFrame()

    def explore_initial(self):
        """Perform an initial round of exploration
        Must be called before explore_batch()
        Returns
        -------
        Optional[float]
            the average score of the batch. None if no objective values were calculated, either due
            to each input failing or no inputs being acquired
        """
        ac = Acquisiton(self.pool)
        sampled_df = ac.random(sample_size=self.initial_size)
        self.pool.drop(index=sampled_df.index, inplace=True)  # update the pool
        self.pool.reset_index(drop=True, inplace=True)
        return sampled_df.reset_index(drop=True)

    def explore_batch(self, model):
        ac = Acquisiton(self.pool)
        # sampled_df = ac.random(sample_size=self.batch_size)
        props = eval_model(model=model, dataset=self.pool)[-1]
        sampled_df = ac.entropy_sample(props=props, sample_size=self.batch_size)
        self.pool.drop(index=sampled_df.index, inplace=True)  # update the pool
        self.pool.reset_index(drop=True, inplace=True)
        return sampled_df.reset_index(drop=True)

    def run(self):
        acc_list = []
        auc_list = []
        chose_data = self.explore_initial()
        self.train_data = pd.concat(
            (self.train_data, chose_data), ignore_index=True
        )  # update training set
        model = Model(
            train_data=self.train_data, epochs=self.epochs
        )  # init model with training set
        fited_model = model.fit()[0]
        acc, auc = eval_model(fited_model, self.test_set)[:2]
        acc_list.append(acc)
        auc_list.append(auc)
        for i in range(self.iters):
            print(i)
            chose_data = self.explore_batch(fited_model)
            self.train_data = pd.concat(
                (self.train_data, chose_data), ignore_index=True
            )
            model = Model(train_data=self.train_data, epochs=self.epochs)
            fited_model = model.fit()[0]
            acc, auc = eval_model(fited_model, self.test_set)[:2]
            acc_list.append(acc)
            auc_list.append(auc)
        return acc_list, auc_list


class Model:
    def __init__(
        self,
        train_data,
        task="acute_toxic",
        batch_size=100,
        epochs=800,
        p_dropout=0.1,
        fingerprint_dim=150,
        radius=3,
        T=2,
        weight_decay=2.9,
        learning_rate=3.5,
        per_task_output_units_num=2,
    ):
        self.train_data = train_data
        self.task = task
        self.batch_size = batch_size
        self.epochs = epochs
        self.p_dropout = p_dropout
        self.fingerprint_dim = fingerprint_dim
        self.radius = radius
        self.T = T
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.per_task_output_units_num = per_task_output_units_num
        self.output_units_num = per_task_output_units_num
        self.weights = [
            train_data.shape[0] / (train_data[task] == 0).shape[0],
            train_data.shape[0] / (train_data[task] == 1).shape[0],
        ]
        self.loss_function = [
            nn.CrossEntropyLoss(torch.Tensor(self.weights), reduction="mean")
        ]
        self.smilesList = train_data.cano_smiles.values
        self.feature_dicts = save_smiles_dicts(self.smilesList, "junk")

        x_atom, x_bonds, _, _, _, _ = get_smiles_array(
            [self.smilesList[0]], self.feature_dicts
        )
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]
        self.model = Fingerprint(
            radius,
            T,
            num_atom_features,
            num_bond_features,
            fingerprint_dim,
            self.output_units_num,
            p_dropout,
        )
        self.model.cuda()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            10**-learning_rate,
            weight_decay=10**-weight_decay,
        )
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])

    def train(self, train_data):
        self.model.train()
        valList = np.arange(0, train_data.shape[0])
        # shuffle them
        np.random.shuffle(valList)
        batch_list = []
        losses_list = []
        for i in range(0, train_data.shape[0], self.batch_size):
            batch = valList[i : i + self.batch_size]
            batch_list.append(batch)
        for counter, train_batch in enumerate(batch_list):
            batch_df = train_data.loc[train_batch, :]
            smiles_list = batch_df.cano_smiles.values

            (
                x_atom,
                x_bonds,
                x_atom_index,
                x_bond_index,
                x_mask,
                smiles_to_rdkit_list,
            ) = get_smiles_array(smiles_list, self.feature_dicts)
            atoms_prediction, mol_prediction = self.model(
                torch.Tensor(x_atom),
                torch.Tensor(x_bonds),
                torch.cuda.LongTensor(x_atom_index),
                torch.cuda.LongTensor(x_bond_index),
                torch.Tensor(x_mask),
            )
            self.model.zero_grad()
            # Step 4. Compute your loss function. (Again, Torch wants the target wrapped in a variable)
            # loss = 0.0
            y_pred = mol_prediction[:, 0:2]

            y_val = batch_df[self.task].values

            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            #             validInds = np.where(y_val != -1)[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss = self.loss_function[0](
                y_pred_adjust, torch.cuda.LongTensor(y_val_adjust)
            )
            # Step 5. Do the backward pass and update the gradient
            #             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)
            losses_list.append(loss.cpu().detach().numpy())
            loss.backward()
            self.optimizer.step()
        return losses_list

    def eval(self, model, dataset):
        batch_size = self.batch_size

        model.eval()
        y_pred_list = []
        predList = np.arange(0, dataset.shape[0])
        batch_list = []

        for i in range(0, dataset.shape[0], batch_size):
            batch = predList[i : i + batch_size]
            batch_list.append(batch)
        for counter, test_batch in enumerate(batch_list):
            batch_df = dataset.loc[test_batch, :]
            smiles_list = batch_df.cano_smiles.values
            feature_dicts = save_smiles_dicts(smiles_list, "junk")
            (
                x_atom,
                x_bonds,
                x_atom_index,
                x_bond_index,
                x_mask,
                smiles_to_rdkit_list,
            ) = get_smiles_array(smiles_list, feature_dicts)
            atoms_prediction, mol_prediction = model(
                torch.Tensor(x_atom),
                torch.Tensor(x_bonds),
                torch.cuda.LongTensor(x_atom_index),
                torch.cuda.LongTensor(x_bond_index),
                torch.Tensor(x_mask),
            )
            # atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()
            y_pred = mol_prediction[:, 0:2]
            y_pred_adjust = F.softmax(y_pred, dim=-1).data.cpu().numpy()[:, 1]

            y_pred_list.append(y_pred_adjust)
        y_pred_list = np.concatenate(y_pred_list)
        y_true = dataset.acute_toxic.values
        acc = accuracy_score(y_true, (y_pred_list > 0.5).astype(int))
        return acc

    # validInds = np.where((y_val == 0) | (y_val == 1))[0]
    # y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
    # validInds = torch.cuda.LongTensor(validInds).squeeze()
    # y_pred_adjust = torch.index_select(y_pred, 0, validInds)
    # loss = self.loss_function[0](y_pred_adjust, torch.cuda.LongTensor(y_val_adjust))

    # y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]

    # #losses_list.append(loss.cpu().detach().numpy())
    # test_roc = [roc_auc_score(y_val_list[0], y_pred_list[0])]
    # test_prc = [
    #     auc(
    #         precision_recall_curve(y_val_list[0], y_pred_list[0])[1],
    #         precision_recall_curve(y_val_list[0], y_pred_list[0])[0],
    #     )
    # ]
    #     test_prc = auc(recall, precision)
    # test_precision = [
    #     precision_score(y_val_list[0], (np.array(y_pred_list[0]) > 0.5).astype(int))
    # ]

    # test_recall = [
    #     recall_score(y_val_list[0], (np.array(y_pred_list[0]) > 0.5).astype(int))
    # ]
    # test_loss = np.array(losses_list).mean()

    # return test_roc, test_prc, test_precision, test_recall, test_loss

    def fit(self, n_splits=3):
        losses = []

        for epoch in range(self.epochs):
            loss = self.train(self.train_data)
            losses.append(sum(loss) / len(loss))
            # _losses = []

            # kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            # for train_index, valid_index in kf.split(
            #     self.train_data["cano_smiles"], self.train_data["acute_toxic"]
            # ):
            #     train_df = self.train_data.iloc[train_index, :]
            #     valid_df = self.train_data.iloc[valid_index, :]
            #     train_df = train_df.reset_index(drop=True)
            #     valid_df = valid_df.reset_index(drop=True)
            #     test_roc, test_prc, test_precision, test_recall, test_loss = self.eval(
            #         valid_df
            #     )
            #     _losses.append(test_roc[0])
            #    self.train(train_df)
            # losses.append(sum(_losses) / len(_losses))
        return self.model, losses


def eval_model(model, dataset, batch_size=100):

    model.eval()
    y_pred_list = []
    predList = np.arange(0, dataset.shape[0])
    batch_list = []

    for i in range(0, dataset.shape[0], batch_size):
        batch = predList[i : i + batch_size]
        batch_list.append(batch)
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.loc[test_batch, :]
        smiles_list = batch_df.cano_smiles.values
        feature_dicts = save_smiles_dicts(smiles_list, "junk")
        (
            x_atom,
            x_bonds,
            x_atom_index,
            x_bond_index,
            x_mask,
            smiles_to_rdkit_list,
        ) = get_smiles_array(smiles_list, feature_dicts)
        atoms_prediction, mol_prediction = model(
            torch.Tensor(x_atom),
            torch.Tensor(x_bonds),
            torch.cuda.LongTensor(x_atom_index),
            torch.cuda.LongTensor(x_bond_index),
            torch.Tensor(x_mask),
        )
        # atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()
        y_pred = mol_prediction[:, 0:2]
        y_pred_adjust = F.softmax(y_pred, dim=-1).data.cpu().numpy()[:, 1]

        y_pred_list.append(y_pred_adjust)
    y_pred_list = np.concatenate(y_pred_list)
    y_true = dataset.acute_toxic.values
    acc = accuracy_score(y_true, (y_pred_list > 0.5).astype(int))
    auc = roc_auc_score(y_true, y_pred_list)
    return acc, auc, y_pred_list
