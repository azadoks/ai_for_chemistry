# %%
import os
# Fall back to CPU on Apple Silicon for unimplemented functions
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pathlib as pl

import pandas as pd
from lightning import pytorch as pl

from chemprop import data, featurizers, nn
from chemprop.nn import metrics
from chemprop.models import multi

# %%
smiles_columns = ['CHROMOPHORE_SMILES', 'SOLVENT_SMILES']
target_columns = ['EMISSION_MAX_NM', 'ABSORPTION_MAX_NM']

df = pd.concat([
    pd.read_json('../data/combined/train.json'),
    pd.read_json('../data/combined/validate.json'),
    pd.read_json('../data/combined/test.json')
]).loc[:, smiles_columns + target_columns]
df = df.where(df != None)
df = df.dropna(axis='index')

df.shape
# %%
X = df.loc[:, smiles_columns].values
y = df.loc[:, target_columns].values
# %%
all_data = [[data.MoleculeDatapoint.from_smi(smis[0], y) for smis, y in zip(X, y)]]
all_data += [[data.MoleculeDatapoint.from_smi(smis[i]) for smis in X] for i in range(1, len(smiles_columns))]

# %%
component_to_split_by = 0 # index of the component to use for structure based splits
mols = [d.mol for d in all_data[component_to_split_by]]
train_indices, val_indices, test_indices = data.make_split_indices(mols=mols, split="random", sizes=(0.8, 0.1, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)
# %%
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_datasets = [data.MoleculeDataset(train_data[i], featurizer) for i in range(len(smiles_columns))]
val_datasets = [data.MoleculeDataset(val_data[i], featurizer) for i in range(len(smiles_columns))]
test_datasets = [data.MoleculeDataset(test_data[i], featurizer) for i in range(len(smiles_columns))]
# %%
train_mcdset = data.MulticomponentDataset(train_datasets)
scaler = train_mcdset.normalize_targets()
val_mcdset = data.MulticomponentDataset(val_datasets)
val_mcdset.normalize_targets(scaler)
test_mcdset = data.MulticomponentDataset(test_datasets)
# %%
train_loader = data.build_dataloader(train_mcdset)
val_loader = data.build_dataloader(val_mcdset, shuffle=False)
test_loader = data.build_dataloader(test_mcdset, shuffle=False)
# %%
mcmp = nn.MulticomponentMessagePassing(
    blocks=[nn.BondMessagePassing() for _ in range(len(smiles_columns))],
    n_components=len(smiles_columns),
)
# %%
agg = nn.MeanAggregation()
# %%
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
# %%
ffn = nn.RegressionFFN(
    n_tasks=2,  # two tasks: EMISSION_MAX_NM and ABSORPTION_MAX_NM
    input_dim=mcmp.output_dim,
    output_transform=output_transform,
    n_layers=1,
    dropout=0.2,
    activation='relu'
)
# %%
metric_list = [metrics.RMSEMetric(), metrics.MAEMetric()] # Only the first metric is used for training and early stopping
# %%
mcmpnn = multi.MulticomponentMPNN(
    mcmp,
    agg,
    ffn,
    metrics=metric_list,
)
# %%
trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=20, # number of epochs to train for
)
# %%
trainer.fit(mcmpnn, train_loader, val_loader)
# %%
