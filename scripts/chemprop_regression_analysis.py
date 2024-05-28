# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('lightning_logs/version_2/metrics.csv')
# %%
fig, ax = plt.subplots()

log = {'epoch': [], 'val_loss': [], 'train_loss': []}
for (epoch, group) in df.groupby('epoch'):
    log['epoch'].append(epoch)
    log['val_loss'].append(group['val_loss'].iloc[-1])
    log['train_loss'].append(group['train_loss'].iloc[-2])

ax.plot(log['epoch'], log['val_loss'], label='Validation Loss')
ax.plot(log['epoch'], log['train_loss'], label='Training Loss')

ax.legend()
# %%
fig, ax = plt.subplots()

train_df = df.dropna(axis='index', subset=['train_loss'])
ax.plot(train_df['step'], train_df['train_loss'], label='Training Loss')

val_df = df.dropna(axis='index', subset=['val_loss'])
ax.plot(val_df['step'], val_df['val_loss'], label='Validation Loss')

ax.legend()
# %%
