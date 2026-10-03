from pathlib import Path
import sys,json,zipfile,xml.etree.ElementTree as ET,collections,hashlib
import numpy as np
import pandas as pd
R=Path(sys.argv[1]).resolve(); RAW=Path(sys.argv[2]).resolve(); D=R/'data';D.mkdir(parents=True,exist_ok=True)
def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
# Read the authoritative strain identifiers without installing another Excel engine.
z=zipfile.ZipFile(RAW/'Data/Strains.xlsx');ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ss=[''.join(v.itertext()) for v in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
rows=[]
for r in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns):
 d={}
 for c in r:
  v=c.find('m:v',ns)
  if v is not None:d[''.join(t for t in c.attrib['r'] if t.isalpha())]=ss[int(v.text)] if c.attrib.get('t')=='s' else v.text
 rows.append(d)
mp={r['R']:r['Q'] for r in rows[1:] if r.get('R') and r.get('Q')};mp.update(RP1_A='R_planticola1',RP1_B='R_planticola1')
author_mp=mp.copy()
mp['Ecoli']=mp['EColi']
mp['EColi_gfp']=mp['Ecoli_gfp']
mp['PAg1_gfp']=mp['PA']
focal_raw={'EC':'EColi_gfp','EA':'EA_gfp','RP1':'RP1_gfp','PAg':'PAg1_gfp','BI':'BI_gfp','CF':'CF_gfp'}
print('focal mapping', {k:mp.get(v) for k,v in focal_raw.items()},flush=True)
assert all(v in mp for v in focal_raw.values())
u=pd.read_csv(RAW/'usable_isos_per_target.csv'); allow={k:set(u[v].dropna()) for k,v in focal_raw.items()}
x=pd.read_csv(RAW/'k2_effects_data.csv')
keep=[a in allow[f] and b in allow[f] and a!=b and 'Mono' not in a+b and c>2 for a,b,f,c in zip(x.sample1,x.sample2,x.focal,x.Count)]
x=x[keep].copy();assert len(x)==5357
x['y24']=x.combined_effect
# Derive 48h with the original well QC and focal monoculture normalization.
monos=['Mono1','Mono2','Mono'];out=[];rebuild_errors=[]
for j,f in enumerate(['EC','EA','RP1','BI','CF','PAg']):
 parts=[]
 for i in [2*j+1,2*j+2]:
  w=pd.read_csv(RAW/f'Data/kChip_data/k2/chip{i}.csv'); mode=w.t0_Area.mode().iloc[0]
  w=w[(w.t0_Area>mode*.7)&(w.t0_Area<mode*1.3)&(w.Total==2)].copy()
  for t in ['t1','t2']:w[t]=(w[t]-w.t0).clip(lower=1)
  parts.append(w)
 w=pd.concat(parts,ignore_index=True)
 mono=w[w.sample1.isin(monos)&w.sample2.isin(monos)]
 denominators={t:float(mono[t].median()) for t in ['t1','t2']}
 w[['sample1','sample2']]=w[['sample1','sample2']].replace({'Blank1':np.nan,'Blank2':np.nan,'Mono1':'Mono','Mono2':'Mono'})
 w['sample1']=w.sample1.fillna(w.sample2);w['sample2']=w.sample2.fillna(w.sample1)
 g=w.groupby(['sample1','sample2'])[['t1','t2']].median();counts=w.groupby(['sample1','sample2'])[['t1','t2']].count()
 for idx,r in x[x.focal==f].iterrows():
  key=(r.sample1,r.sample2)
  if key in g.index:
   rebuild_errors.append(abs(np.log(g.loc[key,'t1']/denominators['t1'])-r.y24))
   if counts.loc[key,'t2']>2 and np.isfinite(g.loc[key,'t2']) and denominators['t2']>0:
    out.append(dict(idx=idx,y48=np.log(g.loc[key,'t2']/denominators['t2'])))
x=x.join(pd.DataFrame(out).set_index('idx')); assert max(rebuild_errors)<1e-6, max(rebuild_errors)
x['aff']=list(zip(x.sample1.map(mp),x.sample2.map(mp)));assert not x.sample1.map(mp).isna().any();assert not x.sample2.map(mp).isna().any()
x['focal_id']=[mp[focal_raw[f]] for f in x.focal]
x['key']=['|'.join(sorted(a)) for a in x.aff]
assert x.groupby(['key','focal_id']).size().max()==1
x['row_id']=['pair_'+str(i) for i in x.index]
# Published trio QC.
t=pd.read_csv(RAW/'k3_effect_data.csv').replace({'RP1':'R_planticola4'})
allowed={author_mp.get(v,v) for v in allow['EC']}
ks=[]
for a,b,c,n in zip(t.sample1,t.sample2,t.sample3,t.Count):ks.append(len({a,b,c})==3 and all(v in allowed for v in [a,b,c]) and n>2)
t=t[ks].copy(); print('trios',len(t),flush=True);assert len(t)==3009
t['aff']=list(zip(t.sample1,t.sample2,t.sample3));t['focal_id']=mp[focal_raw['EC']];t['focal']='EC';t['key']=['|'.join(sorted(a)) for a in t.aff];t['row_id']=['trio_'+str(i) for i in t.index];t['y24']=t.combined_effect
# Full pairs; reuse pilot parent split and coverage moves. Never move pilot-val to test.
parts=['fit','val','test'];moves=[]
OLD=Path(sys.argv[3]).resolve()
robust=R.name=='robust'
def ids_of(d):return set(d.focal_id)|{v for aa in d.aff for v in aa}
def split(df,seed,label):
 parent=pd.read_csv(OLD/'data'/f'{label}_full_split.csv')
 assignments=dict(zip(parent.key,parent.split))
 # Preserve selected pilot coverage moves as well as parent test allocation.
 selected=pd.read_csv(OLD/'data'/('source_selected.csv' if label=='pair' else 'trio_selected.csv'))
 for k,p in zip(selected.key,selected.split):assignments[k]=p
 missing=sorted(set(df.key)-set(assignments))
 for k in missing:
  h=int(hashlib.sha256((str(seed)+k).encode()).hexdigest()[:8],16)/2**32
  assignments[k]='fit' if h<.7 else 'val' if h<.85 else 'test'
 full=df.copy();full['split']=full.key.map(assignments)
 if robust:
  pool=sorted(full[full.split!='test'].key.unique());kk=np.random.default_rng(seed+9000).permutation(pool);nfit=round(len(kk)*.7/.85)
  rr={k:('fit' if i<nfit else 'val') for i,k in enumerate(kk)}
  full['split']=[rr.get(k,p) for k,p in zip(full.key,full.split)]
 while missing:=sorted(ids_of(full)-ids_of(full[full.split=='fit'])):
  v=missing[0];poss=full[(full.focal_id==v)|full.aff.apply(lambda aa:v in aa)]
  poss=poss[poss.split!='test']
  if not len(poss):raise ValueError('Unseen test ID; do not silently move held test')
  k=sorted(poss.key)[0];full.loc[full.key==k,'split']='fit';moves.append(dict(entity=v,key=k))
 full.drop(columns=['aff']).to_csv(D/f'{label}_full_split.csv',index=False)
 return full
s=split(x,1232026,'pair');known=ids_of(s[s.split=='fit'])
t_before=len(t);t=t[t.aff.apply(lambda aa:all(v in known for v in aa))&t.focal_id.isin(known)];tr=split(t,1232027,'trio')
entities=sorted(known);idmap={v:i for i,v in enumerate(entities)}
scales={};sizes={}
for target,df,col in [('source',s,'y24'),('hour48',s,'y48'),('trio',tr,'y24')]:
 df=df[np.isfinite(df[col])].copy();sizes[target]={}
 fit=df[df.split=='fit'];mu=float(fit[col].mean());sd=float(fit[col].std(ddof=0));assert sd>0
 scales[target]={'mean':mu,'std':sd}
 for part in parts:
  d=df[df.split==part];assert len(d)>0
  arr=np.array([[idmap[v] for v in (*sorted(a),f)] for a,f in zip(d.aff,d.focal_id)],np.int32)
  single=np.array([[float(r[f'mono_s{j+1}_effect']) for j in sorted(range(len(r.aff)),key=lambda j:r.aff[j])] for _,r in d.iterrows()],np.float32)
  kwargs=dict(single=single,ids=arr,y=d[col].to_numpy(np.float32),row_id=d.row_id.to_numpy(str),cluster=d.key.to_numpy(str),focal=d.focal.to_numpy(str))
  if target=='source':
   for k in ['strongest','mean','additive']:kwargs[k]=d[k].to_numpy(np.float32)
  if target=='trio':
   for k in ['strongest','mean','additive']:kwargs[k]=d[k+'_single'].to_numpy(np.float32)
  np.savez_compressed(D/f'{target}_{part}.npz',**kwargs);sizes[target][part]=len(d)
 df.drop(columns=['aff']).to_csv(D/f'{target}_selected.csv',index=False)
js(D/'meta.json',dict(entities=entities,n_entities=len(entities),mapping=mp,focal_mapping={k:mp[v] for k,v in focal_raw.items()},scales=scales,sizes=sizes,source_fit_focal_counts=s[s.split=='fit'].focal.value_counts().to_dict()))
js(R/'data_summary.json',dict(full_pair=5357,full_trio_before_known_filter=t_before,full_trio_after_known_filter=len(t),sizes=sizes,n_entities=len(entities),coverage_moves=moves,reconstructed_24h_max_abs_error=max(rebuild_errors),sampling='full groups; pilot parent split with pilot coverage preserved; robust reassigns fit/val only'))
js(D/'strain_mapping.json',mp)
print(json.dumps(json.loads((R/'data_summary.json').read_text()),indent=2),flush=True)
