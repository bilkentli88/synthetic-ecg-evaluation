from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from ddpm_model import ECGDiffusion

p=argparse.ArgumentParser(description='Train/sample the class-conditional 1D DDPM used in the study.')
p.add_argument('--data_dir', required=True); p.add_argument('--out_dir', required=True)
p.add_argument('--seed', type=int, required=True); p.add_argument('--num_classes', type=int, required=True)
p.add_argument('--epochs', type=int, default=120); p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--T', type=int, default=300)
p.add_argument('--base_ch', type=int, default=64); p.add_argument('--ddim_steps', type=int, default=50)
p.add_argument('--guidance', type=float, default=1.0); p.add_argument('--samples_per_class', type=int, default=2000)
p.add_argument('--x0_clip', type=float, default=6.0); p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
D=Path(a.data_dir); O=Path(a.out_dir); O.mkdir(parents=True, exist_ok=True)
X=np.load(D/'X_train.npy').astype(np.float32); y=np.load(D/'y_train.npy').astype(np.int64)
g=ECGDiffusion(num_classes=a.num_classes,T=a.T,base_ch=a.base_ch,device=a.device,seed=a.seed)
g.fit(X,y,epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,seed=a.seed)
g.save_checkpoint(O/f'ddpm_seed{a.seed}.pt',extra=vars(a))
Xs,ys=g.sample_per_class({c:a.samples_per_class for c in range(a.num_classes)},a.ddim_steps,a.guidance,512,True,a.x0_clip)
np.save(O/f'ddpm_X_syn_seed{a.seed}.npy',Xs); np.save(O/f'ddpm_y_syn_seed{a.seed}.npy',ys)
print('saved',O)
