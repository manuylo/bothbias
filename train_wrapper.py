import pandas as pd
import numpy as np
import pickle
import warnings
import os
import argparse
from rdkit.Chem import Descriptors
from pymol import cmd
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import reduce
from lightgbm import LGBMRegressor
from rdkit import Chem, RDLogger

RDLogger.DisableLog("*")
warnings.filterwarnings("ignore")

AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [
        os.path.join(data_dir, protein_file) for protein_file in df["protein"]
    ]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df["ligand"]]
    keys = df["key"]
    pks = df["pk"]
    return protein_files, ligand_files, keys, pks


def get_active_site_amino_acids(csv_file, pdb, protein_file, ligand_file, cutoff):
    if not os.path.exists(f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}"):
        os.makedirs(f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}")
    cmd.load(protein_file, pdb)
    cmd.load(ligand_file, pdb + "_ligand")
    cmd.select(f"{pdb}_active_site", f"byres {pdb} within {cutoff} of {pdb}_ligand")
    cmd.create(pdb + "_active_site", f"{pdb}_active_site")
    cmd.save(
        "data/scratch/"
        + f"{csv_file.split('/')[-1].split('.')[0]}/"
        + f"{pdb}_active_site_{cutoff}.pdb",
        pdb + "_active_site",
    )
    cmd.delete(pdb)
    cmd.delete(pdb + "_ligand")
    cmd.delete(pdb + "_active_site")
    with open(
        "data/scratch/"
        + f"{csv_file.split('/')[-1].split('.')[0]}/"
        + f"{pdb}_active_site_{cutoff}.pdb",
        "r",
    ) as f:
        lines = f.readlines()
    residues_nums = []
    for line in lines:
        if line.startswith("ATOM"):
            residues_nums.append(
                (int(line[22:26]), line[21:22].strip(), line[17:20].strip())
            )
    residues_nums = list(set(residues_nums))
    cleaned_residue_nums = residues_nums
    return cleaned_residue_nums


def simple_featurise_amino_acid_sequence(
    csv_file, pdb, protein_file, ligand_file, cutoff
):
    residues_nums = get_active_site_amino_acids(
        csv_file, pdb, protein_file, ligand_file, cutoff=cutoff
    )
    amino_acid_counts = {amino_acid: 0 for amino_acid in AMINO_ACIDS}
    for residue in residues_nums:
        amino_acid_counts[residue[2]] += 1
    os.system(
        "rm data/scratch/"
        + f"{csv_file.split('/')[-1].split('.')[0]}/"
        + f"{pdb}_active_site_{cutoff}.pdb"
    )
    return {pdb: list(amino_acid_counts.values())}


def featurise_data_simple_active_site(csv_file, data_dir, cutoff):
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    protein_files, ligand_files, pdbs, _ = load_csv(csv_file, data_dir)
    all_embeddings = {}
    with Parallel(n_jobs=8) as parallel:
        results = parallel(
            delayed(simple_featurise_amino_acid_sequence)(
                csv_file, pdb, protein_file, ligand_file, cutoff=cutoff
            )
            for pdb, protein_file, ligand_file in tqdm(
                zip(pdbs, protein_files, ligand_files)
            )
        )
    all_embeddings = pd.DataFrame(reduce(lambda r, d: r.update(d) or r, results, {})).T
    all_embeddings.to_csv(
        "data/features/"
        + csv_file.split("/")[-1].split(".")[0]
        + f"_protein_bias_features_{cutoff}.csv"
    )
    return None


def train_combined_active_site_model(csv_file, data_dir, cutoff=15):
    model = LGBMRegressor(
        n_estimators=205,
        num_leaves=291,
        min_child_samples=2,
        learning_rate=0.03295785797670332,
        log_max_bin=9,
        colsample_bytree=0.5813381312278044,
        reg_alpha=0.011572343074847936,
        reg_lambda=0.011739705334667914,
        n_jobs=-1,
    )
    protein_features = pd.read_csv(
        "data/features/"
        + csv_file.split("/")[-1].split(".")[0]
        + f"_protein_bias_features_{cutoff}.csv",
        index_col=0,
    )
    ligand_features = pd.read_csv(
        "data/features/"
        + csv_file.split("/")[-1].split(".")[0]
        + "_ligand_bias_features.csv",
        index_col=0,
    )
    all_embeddings = pd.concat([protein_features, ligand_features], axis=1)
    labels = pd.read_csv(csv_file)["pk"].to_list()
    model.fit(all_embeddings, labels)
    return model


def predict_combined_active_site_model(model_name, csv_file, data_dir, cutoff=15):
    model = pickle.load(
        open(
            f"data/models/{model_name}.pkl",
            "rb",
        )
    )
    protein_features = pd.read_csv(
        "data/features/"
        + csv_file.split("/")[-1].split(".")[0]
        + f"_protein_bias_features_{cutoff}.csv",
        index_col=0,
    )
    ligand_features = pd.read_csv(
        "data/features/"
        + csv_file.split("/")[-1].split(".")[0]
        + "_ligand_bias_features.csv",
        index_col=0,
    )
    # combine ligand and protein embeddings
    all_embeddings = pd.concat([protein_features, ligand_features], axis=1)
    predictions = model.predict(all_embeddings)
    _, _, keys, pks = load_csv(csv_file, data_dir)
    return pd.DataFrame({"key": keys, "pred": predictions, "pk": pks})


def read_pdb_line(line):
    chain_id = line[21:22].strip()
    res_number = line[22:26].strip()
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    return x, y, z, res_number, chain_id


def generate_rdkit_features(csv_file, data_dir):
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    feature_names = get_rdkit_features_names()
    _, ligand_files, keys, _ = load_csv(csv_file, data_dir)
    with Parallel(n_jobs=-1) as parallel:
        results = parallel(
            delayed(get_rdkit_features)(ligand_file, feature_names)
            for ligand_file in tqdm(ligand_files)
        )
    features = {}
    for pdb, result in zip(keys, results):
        features[pdb] = [result[feature_name] for feature_name in feature_names]
    data = pd.DataFrame(features, index=feature_names).T
    data.to_csv(
        "data/features/"
        + f"{csv_file.split('/')[-1].split('.')[0]}"
        + "_ligand_bias_features.csv"
    )
    return None


def get_rdkit_features_names():
    with open("rdkit_feature_names.txt", "r") as f:
        return f.read().splitlines()


def get_rdkit_features(ligand_file, feature_names):
    descriptors = {d[0]: d[1] for d in Descriptors.descList}
    mol = Chem.MolFromMolFile(ligand_file, removeHs=False)
    features = {}
    for feature_name in feature_names:
        try:
            features[feature_name] = descriptors[feature_name](mol)
        except:
            features[feature_name] = np.nan
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--val_csv_file", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--cutoff", type=float, default=15)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    args = parser.parse_args()
    if args.train:
        if not os.path.exists(
            "data/features/"
            + f"{args.csv_file.split('/')[-1].split('.')[0]}"
            + "_ligand_bias_features.csv"
        ):
            generate_rdkit_features(args.csv_file, args.data_dir)
        if not os.path.exists(
            "data/features/"
            + f"{args.csv_file.split('/')[-1].split('.')[0]}"
            + f"_protein_bias_features_{args.cutoff}.csv"
        ):
            featurise_data_simple_active_site(args.csv_file, args.data_dir, args.cutoff)
        model = train_combined_active_site_model(
            args.csv_file, args.data_dir, args.cutoff
        )
        if not os.path.exists("data/models"):
            os.makedirs("data/models")
        with open(f"data/models/{args.model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    elif args.predict:
        if not os.path.exists(
            "data/features/"
            + f"{args.val_csv_file.split('/')[-1].split('.')[0]}"
            + "_ligand_bias_features.csv"
        ):
            generate_rdkit_features(args.val_csv_file, args.val_data_dir)
        if not os.path.exists(
            "data/features/"
            + f"{args.val_csv_file.split('/')[-1].split('.')[0]}"
            + f"_protein_bias_features_{args.cutoff}.csv"
        ):
            featurise_data_simple_active_site(
                args.val_csv_file, args.val_data_dir, args.cutoff
            )
        df = predict_combined_active_site_model(
            args.model_name,
            args.val_csv_file,
            args.val_data_dir,
            args.cutoff,
        )
        if not os.path.exists("data/results"):
            os.makedirs("data/results")
        df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}.csv',
            index=False,
        )
    else:
        raise ValueError("Please specify either --train or --predict")
