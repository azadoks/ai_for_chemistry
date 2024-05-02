# %%
import pandas as pd
from urllib.request import urlopen
from urllib.parse import quote

PATH = '../data/chemfluor/raw_data.xlsx'
# %%
def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        smiles = urlopen(url).read().decode('utf8')
        return smiles
    except Exception:
        return None
# %%
df = pd.read_excel(PATH, index_col=0)
# %%
solvent_smiles = {}
for solvent in df.solvent.unique():
    smiles = CIRconvert(solvent)
    if smiles is not None:
        solvent_smiles[solvent] = smiles
# %%
df['solvent_smiles'] = df['solvent'].apply(lambda x: solvent_smiles.get(x, pd.NA))
df.to_csv('../data/chemfluor/data.csv', index=False)
