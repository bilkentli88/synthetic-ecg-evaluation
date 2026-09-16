from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
metrics=['DistributionalMMD2','Diversity','StabilityError','UtilityAccuracy','MacroF1','ClassRecallBalance','PhysiologicalValidity']
for ds in ['mitbih','svdb']:
    p=root/'results'/f'{ds}_seed_results.csv'
    df=pd.read_csv(p)
    assert set(df.Model)=={'TimeGAN','LSTM-VAE','DDPM'}
    assert set(df.Seed)=={19,88,123}
    assert len(df)==9 and not df[metrics].isna().any().any()
    print('\n',ds.upper())
    print(df.groupby('Model')[metrics].agg(['mean','std']).round(6))
print('\nPASS: checked 18 raw result rows; no missing metrics.')
