"""Evaluate selected full-data models and compute paired cluster comparisons."""
from pathlib import Path
import json,sys,importlib.util,itertools
import numpy as np,joblib
ROOT=Path(__file__).resolve().parent;R=Path(sys.argv[1]).resolve();sys.path.insert(0,str(R));import model as m
spec=importlib.util.spec_from_file_location('bench',ROOT/'baselines.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
O=R/'evaluation';O.mkdir(exist_ok=True);P=m.P;M=m.M

def metric(y,p,f):
 p=np.atleast_2d(p);err=(p-y)**2;by={str(k):float(err[:,f==k].mean()) for k in np.unique(f)}
 return dict(mse=float(err.mean()),seed_mse=err.mean(1).tolist(),rmse=float(np.sqrt(err.mean())),mae=float(np.abs(p-y).mean()),by_focal=by,macro_mse=float(np.mean(list(by.values()))))
def paired(z,a,b,label,family):
 a=np.atleast_2d(a);b=np.atleast_2d(b);ok=np.isfinite(a).all(0)&np.isfinite(b).all(0);y=z['y'][ok];da=(a[:,ok]-y)**2-(b[:,ok]-y)**2;d=da.mean(0)
 keys,ix=np.unique(z['cluster'][ok],return_inverse=True);s=np.bincount(ix,weights=d);n=np.bincount(ix);rng=np.random.default_rng(1242026);v=[]
 for _ in range(200):
  draw=rng.integers(len(n),size=(100,len(n)));v.extend((s[draw].sum(1)/n[draw].sum(1)).tolist())
 bm=float(np.mean((b[:,ok]-y)**2));f=z['focal'][ok]
 return dict(label=label,rows=len(y),clusters=len(n),difference=float(d.mean()),improvement_percent=100*float(-d.mean())/bm,seed_differences=da.mean(1).tolist(),ci95=np.quantile(v,[.025,.975]).tolist(),family_size=family,family_ci=np.quantile(v,[.025/family,1-.025/family]).tolist(),leave_one_focal_out={str(k):float(d[f!=k].mean()) for k in np.unique(f) if (f!=k).any()})
def neural(stage,rep,seed,task,z):
 out=R/'runs'/f'{stage}_{task}_{rep}_{seed}'
 if stage=='source':obj=m.Finder(seed,rep)
 else:
  u=np.load(out/'initial_latents.npz')['latent'];obj=(m.AugReader if stage=='aug' else m.Reader)(u,z['ids'].shape[1],seed)
  if stage=='aug':
   sc=json.loads((out/'extra_scale.json').read_text());z=dict(z);z['single_std']=((z['single']-sc['mean'])/sc['std']).astype('float32')
 obj.ck.restore(str(out/'best_state')).expect_partial();p=obj.predict(z);sc=M['scales'][task];p=p*sc['std']+sc['mean'];return p.mean(1) if stage=='source' else p

def run():
 allcom=list(ROOT.glob('*/runs/*/complete.json'));assert len(allcom)==35,len(allcom)
 unresolved=[str(p) for p in allcom if json.loads(p.read_text())['budget_unresolved']]
 if unresolved:raise RuntimeError('TEST HELD: '+str(unresolved))
 assert all((ROOT/s/'baselines/complete.json').exists() for s in ['primary','robust'])
 results={};comparisons=[]
 for part in ['val','test']:
  results[part]={}
  for task in ['source','hour48','trio']:
   z=dict(np.load(R/'data'/f'{task}_{part}.npz'));fit=m.load(task,'fit');out=R/'baselines'/task;pred={}
   pred['constant']=np.array([fit['y'][fit['focal']==f].mean() for f in z['focal']])
   qs=joblib.load(out/'ridge.joblib');x=b.design(z,'rf',M);n=M['n_entities'];pr=np.zeros(len(z['y']))
   for f,q in qs.items():
    ii=z['focal']==f
    if ii.any():pr[ii]=q.predict(x[ii,:n])
   pred['ridge']=pr;pred['rf']=np.stack([q.predict(x) for q in joblib.load(out/'rf.joblib')])
   pred['xgb']=np.stack([q.predict(x) for q in joblib.load(out/'xgb.joblib')])
   for rep in ['gm']:
    stage='source' if task=='source' else 'reader';pred[rep]=np.stack([neural(stage,rep,s,task,z) for s in P['seeds']])
    if stage=='reader':
     pr=[]
     for s in P['seeds']:
      u=np.load(R/'runs'/f'reader_{task}_{rep}_{s}'/'initial_latents.npz')['latent'];xx=np.concatenate([u[z['ids'][:,:-1]].sum(1),u[z['ids'][:,-1]]],axis=1);pr.append(joblib.load(R/'baselines'/f'linear_{task}_{rep}_{s}.joblib').predict(xx))
     pred['linear_'+rep]=np.stack(pr)
   for name in ['strongest','mean','additive','od_weighted']:pred[name]=z[name]
   if task=='source' and R.name=='primary':
    for rep in ['gm']:pred['aug_'+rep]=np.stack([neural('aug',rep,s,'source',z) for s in P['seeds']])
    qs=joblib.load(R/'baselines/extra_rf.joblib');pred['extra_rf']=np.stack([q.predict(b.extra_design(z,M)) for q in qs])
    pred['extra_xgb']=np.stack([q.predict(b.extra_design(z,M)) for q in joblib.load(R/'baselines/extra_xgb.joblib')])
   mm={}
   for name,p in pred.items():
    ok=np.isfinite(np.atleast_2d(p)).all(0);mm[name]=dict(rows=int(ok.sum()),**metric(z['y'][ok],np.atleast_2d(p)[:,ok],z['focal'][ok]))
   results[part][task]=mm;np.savez_compressed(O/f'{task}_{part}_predictions.npz',**{k:z[k] for k in ['y','row_id','cluster','focal']},**pred)
   if part=='test':
    if task in ['source','hour48']:
     for other in ['ridge','rf','xgb']+(['strongest'] if task=='source' else []):
      comparisons.append(paired(z,pred['gm'],pred[other],task+' CDLD - '+other,12))
    if task=='source' and R.name=='primary':
     for other in ['strongest','extra_xgb']:
      comparisons.append(paired(z,pred['aug_gm'],pred[other],'extra CDLD - '+other,4))
 m.js(O/'results.json',results);m.js(O/'comparisons.json',comparisons);print(json.dumps(comparisons,indent=2),flush=True)
if __name__=='__main__':run()
