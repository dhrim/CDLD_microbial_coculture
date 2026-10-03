"""Reconstruct original measured-effect rules without training or test scoring."""
from pathlib import Path
import json,urllib.request,hashlib
import numpy as np,pandas as pd
import sys
R=Path(sys.argv[1]).resolve();RAW=Path(sys.argv[2]).resolve();p=RAW/'Data/Isolate_profiling/gc_data.csv'
gc=pd.read_csv(p);od=dict(zip(gc['sample'],gc['max_od']))
for split in ['primary','robust']:
 S=R/split;M=json.loads((S/'data/meta.json').read_text());mp=M['mapping'];singles48={}
 for j,f in enumerate(['EC','EA','RP1','BI','CF','PAg']):
  parts=[]
  for i in [2*j+1,2*j+2]:
   w=pd.read_csv(RAW/f'Data/kChip_data/k2/chip{i}.csv');mode=w.t0_Area.mode().iloc[0];w=w[(w.t0_Area>mode*.7)&(w.t0_Area<mode*1.3)&(w.Total==2)].copy();w['t2']=(w.t2-w.t0).clip(lower=1);parts.append(w)
  w=pd.concat(parts);mono=w[w.sample1.isin(['Mono','Mono1','Mono2'])&w.sample2.isin(['Mono','Mono1','Mono2'])];den=mono.t2.median()
  w[['sample1','sample2']]=w[['sample1','sample2']].replace({'Blank1':np.nan,'Blank2':np.nan,'Mono1':'Mono','Mono2':'Mono'});w['sample1']=w.sample1.fillna(w.sample2);w['sample2']=w.sample2.fillna(w.sample1)
  g=w.groupby(['sample1','sample2']).t2.agg(['median','count'])
  # Match author single definition: duplicated strain or one monoculture droplet.
  for (a,b),v in g.iterrows():
   if a==b or 'Mono' in [a,b]:
    name=b if a=='Mono' else a
    if name in mp and v['count']>2 and np.isfinite(v['median']):singles48.setdefault((f,mp[name]),[]).append(float(np.log(v['median']/den)))
 singles48={k:float(np.mean(v)) for k,v in singles48.items()}
 for task in ['source','hour48','trio']:
  for part in ['fit','val','test']:
   path=S/'data'/f'{task}_{part}.npz';z=dict(np.load(path));k=z['ids'].shape[1]-1
   if task=='hour48':z['single']=np.array([[singles48.get((str(f),M['entities'][int(i)]),np.nan) for i in ids[:-1]] for f,ids in zip(z['focal'],z['ids'])],np.float32)
   e=z['single'];ok=np.isfinite(e).all(1);strong=np.full(len(e),np.nan,np.float32)
   strong[ok]=e[ok][np.arange(ok.sum()),np.argmax(abs(e[ok]),axis=1)]
   # Keep exact supplied 24h rules and validate reproduction.
   if task!='hour48':
    for name,v in [('strongest',strong),('mean',e.mean(1)),('additive',e.sum(1))]:assert np.allclose(z[name],v,atol=2e-6,equal_nan=True),(task,name)
   else:z.update(strongest=strong,mean=e.mean(1),additive=e.sum(1))
   w=np.array([[od.get(M['entities'][int(i)],np.nan) for i in ids[:-1]] for ids in z['ids']]);z['od_weighted']=(np.sum(w*e,axis=1)/np.sum(w,axis=1)).astype('float32')
   np.savez_compressed(path,**z)
print('Measured rules constructed; test metrics not evaluated',flush=True)
