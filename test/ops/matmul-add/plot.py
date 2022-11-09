#!/bin/bash

#%% 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams['font.size'] = 8 

setups = ['O0', 'O2']
metrics = ['Front', 'BS', 'MEM', 'CORE', 'Retire']

fig, axes = plt.subplots(nrows=1, ncols=len(setups))

# read original data
dfs = [pd.read_csv(f"{s}.csv") for s in setups]
dfs = [df.rename(columns={'Workload': ''}).set_index('') for df in dfs]

# TMA metrics plots
for i in range(len(dfs)):
    df = dfs[i]
    df = df[[*metrics]]
    df.plot(ax=axes[i], kind="barh", stacked=True, legend=False, title=setups[i])

plt.legend(bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()

# TMA data table
tdfs = [df[[*metrics]] for df in dfs]
df = pd.concat(tdfs, axis=1)
df.columns = pd.MultiIndex.from_product([setups, metrics])
display(df)

# TMA data table
tdfs = [df[['Cycles', 'IPC']] for df in dfs]
df = pd.concat(tdfs, axis=1)
df.columns = pd.MultiIndex.from_product([setups, ['Cycles', 'IPC']])
df['O2','Up'] = df['O0','Cycles'] / df['O2','Cycles']
df.style.apply(lambda x: ['background: lightblue' for i in x])
#df['O2'].style.background_gradient(subset=pd.IndexSlice[:, pd.IndexSlice[:, 'Up']])
df
# %%
