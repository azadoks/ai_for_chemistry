# %%
import openai

from loguru import logger

logger.enable("gptchem")
from pathlib import Path

import pandas as pd
import selfies
from ai4chem.data import ChemFluorDataset, Deep4ChemDataset
from fastcore.xtras import save_pickle # type: ignore
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdinchi

from gptchem.baselines.bandgap import train_test_bandgap_regression_baseline
from gptchem.evaluator import get_regression_metrics
from gptchem.extractor import RegressionExtractor
from gptchem.formatter import RegressionFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

# %%
def get_photo_data():
    chemfluor = ChemFluorDataset("../data/chemfluor/data.csv", canonicalize_smiles=False)
    chemfluor_df = chemfluor.raw_data
    chemfluor_df = chemfluor_df.rename({
        chemfluor._chromophore_smiles_column: "SMILES",
        chemfluor._solvent_smiles_column: "SOLVENT_SMILES",
        chemfluor._emission_max_column: "EMISSION_MAX_NM",
        chemfluor._absorption_max_column: "ABSORPTION_MAX_NM",
    }, axis='columns')

    deep4chem = Deep4ChemDataset("../data/deep4chem/data.csv", canonicalize_smiles=False)
    deep4chem_df = deep4chem.raw_data
    deep4chem_df = deep4chem_df.rename({
        deep4chem._chromophore_smiles_column: "SMILES",
        deep4chem._solvent_smiles_column: "SOLVENT_SMILES",
        deep4chem._emission_max_column: "EMISSION_MAX_NM",
        deep4chem._absorption_max_column: "ABSORPTION_MAX_NM",
    }, axis='columns')

    df = pd.concat(
        [
            chemfluor_df[['SMILES', 'SOLVENT_SMILES', 'EMISSION_MAX_NM', 'ABSORPTION_MAX_NM']],
            deep4chem_df[['SMILES', 'SOLVENT_SMILES', 'EMISSION_MAX_NM', 'ABSORPTION_MAX_NM']],
        ],
    axis=0)

    df = df.dropna(axis='index')

    df['SMILES'] = df['SMILES'].apply(Chem.CanonSmiles)
    df['SOLVENT_SMILES'] = df['SMILES'].apply(Chem.CanonSmiles)

    df['SELFIES'] = df['SMILES'].apply(lambda x: selfies.encoder(x, strict=False))
    df['SOLVENT_SELFIES'] = df['SOLVENT_SMILES'].apply(lambda x: selfies.encoder(x, strict=False))

    df['INCHI'] = df['SMILES'].apply(lambda x: rdinchi.MolToInchi(Chem.MolFromSmiles(x)))
    df['SOLVENT_INCHI'] = df['SOLVENT_SMILES'].apply(lambda x: rdinchi.MolToInchi(Chem.MolFromSmiles(x)))

    return df

df = get_photo_data()
# %%
def train_test_model(representation, property_name, num_train_points, seed):
    data = get_photo_data()
    bins = data[property_name] > data[property_name].median()

    train_data, test_data = train_test_split(
        data,
        train_size=num_train_points,
        test_size=min((max_num_test_points, len(data) - num_train_points)),
        stratify=bins,
        random_state=seed,
    )

    # train_smiles = train_data["SMILES"].values
    # test_smiles = test_data["SMILES"].values

    formatter = RegressionFormatter(
        representation_column=representation,
        property_name=property_name,
        label_column=property_name,
    )

    train_formatted = formatter(train_data)
    test_formatted = formatter(test_data)

    # gpr_baseline = train_test_bandgap_regression_baseline(
    #     data, train_smiles=train_smiles, test_smiles=test_smiles, formatter=formatter
    # )

    tuner = Tuner(base_model="gpt-3.5-turbo", n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    tune_res = tuner(train_formatted)
    querier = Querier(tune_res["model_name"])
    completions = querier(test_formatted)
    extractor = RegressionExtractor()
    extracted = extractor(completions)

    res = get_regression_metrics(test_formatted["label"].values, extracted)

    summary = {
        "representation": representation,
        "property_name": property_name,
        "num_train_points": num_train_points,
        **res,
        # "gpr_baseline": gpr_baseline,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)

    print(
        f"Ran train size {num_train_points} and got MAE {res['mean_absolute_error']}"
        # f", GPR baseline {gpr_baseline['mean_absolute_error']}"
    )
# %%
num_training_points = [1000] # [10, 20, 50, 100, 200, 1000, 5000][::-1]  # 1000
representations = ["SMILES"]  # ["SMILES", "SELFIES", "INCHI"]
property_names = ["EMISSION_MAX_NM"]  # ["EMISSION_MAX_NM", "ABSORPTION_MAX_NM"]
max_num_test_points = 250
num_repeats = 1  # 10
# %%
if __name__ == "__main__":
    for seed in range(num_repeats):
        for property_name in property_names:
            for representation in representations:
                for num_train_points in num_training_points:
                    train_test_model(representation, property_name, num_train_points, seed + 165)

# %%
