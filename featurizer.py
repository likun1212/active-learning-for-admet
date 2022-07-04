# -*- encoding: utf-8 -*-
"""
@File    :   featurizer.py
@Time    :   2022/07/01 15:04:21
@Author  :   likun.yang 
@Contact :   likun_yang@foxmail.com
@Description: 
"""

"""A featurizer transforms input representations into uncompressed feature representations for use
with clustering and model training/prediction."""

import numpy as np
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray


class Featurizer:
    def __init__(self, fingerprint: str = "pair", radius: int = 2, length: int = 2048):
        self.fingerprint = fingerprint
        self.radius = radius
        self.length = length

    def __len__(self):
        return self.length

    def __call__(self, smi: str):
        return featurize(smi, self.fingerprint, self.radius, self.length)


def featurize(smi, fingerprint, radius, length):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fingerprint == "morgan":
        fp = rdmd.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=length, useChirality=True
        )
    elif fingerprint == "pair":
        fp = rdmd.GetHashedAtomPairFingerprintAsBitVect(
            mol, minLength=1, maxLength=1 + radius, nBits=length
        )
    elif fingerprint == "rdkit":
        fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=1 + radius, fpSize=length)
    elif fingerprint == "maccs":
        fp = rdmd.GetMACCSKeysFingerprint(mol)
    elif fingerprint == "map4":
        fp = map4.MAP4Calculator(
            dimensions=length, radius=radius, is_folded=True
        ).calculate(mol)
    else:
        raise NotImplementedError(f'Unrecognized fingerprint: "{fingerprint}"')

    X = np.empty(len(fp))
    ConvertToNumpyArray(fp, X)
    return X
