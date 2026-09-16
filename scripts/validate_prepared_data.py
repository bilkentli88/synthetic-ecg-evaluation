from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--dataset',choices=['mitbih','svdb'],required=True)
p.add_argument('--prepared_root',required=True)
p.add_argument('--counts_csv',default=str(Path(__file__).resolve().parents[1]/'splits'/'prepared_class_counts.csv'))
a=p.parse_args()
expected=pd.read_csv(a.counts_csv)
expected=expected[expected.Dataset==a.dataset]
labels=['N','S','V','F','Q'] if a.dataset=='mitbih' else ['N','S','V']
for seed in [19,88,123]:
    d=Path(a.prepared_root)/f'seed_{seed}'
    for split in ['train','test']:
        X=np.load(d/f'X_{split}.npy'); y=np.load(d/f'y_{split}.npy').astype(int)
        assert X.shape[0]==len(y) and X.shape[-1]==256, (seed,split,X.shape,y.shape)
        exp=expected[(expected.Seed==seed)&(expected.Split==split)]
        got={labels[i]:int((y==i).sum()) for i in range(len(labels))}
        want={r.Class:int(r.Count) for _,r in exp.iterrows()}
        print(a.dataset,seed,split,X.shape,got)
        assert got==want, f'Count mismatch: got={got}, expected={want}'
print('PASS: prepared arrays match the recorded experiment class counts.')
