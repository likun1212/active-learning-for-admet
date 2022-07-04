import pandas as pd
from rdkit import Chem

from AttentiveFP import save_smiles_dicts


def process_data(path):
    raw_filename = path
    filename = raw_filename.replace(".csv", "")
    prefix_filename = raw_filename.split("/")[-1].replace(".csv", "")
    smiles_tasks_df = pd.read_csv(
        raw_filename
    )  # if use under_sampling, need comment this line
    smilesList = smiles_tasks_df.smiles.values
    # print("number of all smiles: ",len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
            )
        except:
            # print("not successfully processed smiles: ", smiles)
            # sys.stdout.flush()
            pass
    # print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
    # print(smiles_tasks_df)
    smiles_tasks_df["cano_smiles"] = canonical_smiles_list
    assert canonical_smiles_list[8] == Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles_tasks_df["cano_smiles"][8]),
        isomericSmiles=True,
    )

    smilesList = [
        smiles
        for smiles in canonical_smiles_list
        if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 101
    ]
    uncovered = [
        smiles
        for smiles in canonical_smiles_list
        if len(Chem.MolFromSmiles(smiles).GetAtoms()) > 100
    ]

    smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)]
    feature_dicts = save_smiles_dicts(smilesList, filename)

    remained_df = smiles_tasks_df[
        smiles_tasks_df["cano_smiles"].isin(feature_dicts["smiles_to_atom_mask"].keys())
    ]
    return remained_df, smilesList, feature_dicts
