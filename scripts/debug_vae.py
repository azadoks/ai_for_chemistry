# %%
import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split

from ai4chem.tokenizers import Tokenizer, schwaller_smiles_regex, get_vocabulary
from ai4chem.vae_torch import VAE
from ai4chem.vae_torch import loss as vae_loss
from ai4chem.data import Deep4ChemDataset

# %%
dataset = Deep4ChemDataset('../data/deep4chem/data.csv')
df = dataset.clean_data
df['combined_smiles'] = df.chromophore_smiles + '.' + df.solvent_smiles
# %%
tokenizer = Tokenizer(get_vocabulary(df.combined_smiles.tolist(), tokenizer=schwaller_smiles_regex))
# %%
pad_to_length = df['combined_smiles'].apply(lambda x: len(schwaller_smiles_regex(x))).max()
df['tokenized_combined_smiles'] = df['combined_smiles'].apply(
    lambda x: tokenizer.encode(x, tokenizer=schwaller_smiles_regex, pad_to_length=pad_to_length)
)

X = torch.tensor(df['tokenized_combined_smiles'].tolist())
X = F.one_hot(X, num_classes=len(tokenizer.vocabulary))
X = X.to(dtype=torch.float32)
# %%
X_train, X_test = train_test_split(X, train_size=0.2, random_state=42)

data_train = torch.utils.data.TensorDataset(X_train)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True)

torch.manual_seed(42)

epochs = 5
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model = VAE(embedding_dim=X.shape[1], latent_dim=292, vocabulary_size=len(tokenizer.vocabulary)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        output, mean, logvar = model.forward(data)

        if batch_idx==0:
              inp = data.cpu().numpy()
              outp = output.cpu().detach().numpy()
              print("Input:")
              print(tokenizer.decode(inp[0].argmax(axis=1).tolist()))
              print("Output:")
              print(tokenizer.decode(outp[0].argmax(axis=1).tolist()))

        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        train_loss += loss
        optimizer.step()

    print('train', train_loss / len(train_loader.dataset))
    return train_loss / len(train_loader.dataset)

# %%
%%time
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
# %%
