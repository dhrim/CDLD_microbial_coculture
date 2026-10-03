"""Test access occurs only after all neural/baseline model selection completes."""
import json,sys,itertools
from pathlib import Path
import numpy as np
import joblib
R=Path(__file__).resolve().parent;OLD=R.parent/'full/primary';sys.path.insert(0,str(OLD.parent/'vendor'));sys.path.insert(0,str(OLD));import model as m
P=json.loads((R/'plan.json').read_text());items=json.loads((R/'subsets.json').read_text());O=R/'evaluation';O.mkdir(exist_ok=True)
for x in items:
 for c in P['conditions']:
  for s in P['seeds']:
   run=R/'subsets'/x['name']/c/'runs'/f'reader_hour48_initial_{s}'
   assert not json.loads((run/'complete.json').read_text())['budget_unresolved']
 if x['pct']!=100:assert (R/'subsets'/x['name']/'baselines/complete.json').exists()
test=dict(np.load(OLD/'data/hour48_test.npz'));y=test['y'];n=m.M['n_entities'];aff=np.zeros((len(y),n))
for j in range(2):aff[np.arange(len(y)),test['ids'][:,j]]+=1
features=np.concatenate([aff,np.eye(n)[test['ids'][:,-1]]],1).astype('float32');records=[];losses={};timing=[]
for x in items:
 base=R/'subsets'/x['name'];sc=x['scale'];fit=dict(np.load(base/'data/hour48_fit.npz'));preds={}
 for c in P['conditions']:
  pp=[]
  for s in P['seeds']:
   run=base/c/'runs'/f'reader_hour48_initial_{s}';u=np.load(run/'best_latents.npz')['latent'];obj=m.Reader(u,3,s);obj.ck.restore(str(run/'best_state')).expect_partial();pr=obj.predict(test)*sc['std']+sc['mean'];assert np.isfinite(pr).all();pp.append(pr)
   for stage,rep in [('reader','initial')]+([('source','gm')] if c.endswith('U') else []):
    info=json.loads((base/c/'runs'/f'{stage}_hour48_{rep}_{s}'/'complete.json').read_text());timing.append(dict(subset=x['name'],condition=c,seed=s,stage=stage,seconds=info['seconds'],steps=info['steps'],selected=info['selected']['step'],fit=info['selected']['fit'],val=info['selected']['val']))
  preds[c]=np.array(pp)
 bd=OLD/'baselines/hour48' if x['pct']==100 else base/'baselines/hour48'
 models=joblib.load(bd/'ridge.joblib');pr=np.full(len(y),float(fit['y'].mean()))
 for f,q in models.items():
  ix=test['focal']==f
  if ix.any():pr[ix]=q.predict(features[ix,:n])
 preds['ridge']=np.repeat(pr[None],5,0)
 for label in ['rf','xgb']:preds[label]=np.array([q.predict(features) for q in joblib.load(bd/(label+'.joblib'))])
 means={str(f):float(fit['y'][fit['focal']==f].mean()) for f in np.unique(fit['focal'])};pr=np.array([means.get(str(f),float(fit['y'].mean())) for f in test['focal']]);preds['constant']=np.repeat(pr[None],5,0)
 np.savez_compressed(O/(x['name']+'_predictions.npz'),**preds,y=y,row_id=test['row_id'],cluster=test['cluster'],focal=test['focal'])
 for label,pr in preds.items():
  loss=(pr-y)**2;losses[(x['pct'],x['repeat'],label)]=loss
  records.append(dict(subset=x['name'],pct=x['pct'],repeat=x['repeat'],model=label,mse=float(loss.mean()),seed_mse=loss.mean(1).tolist(),focal_mse={str(f):float(loss[:,test['focal']==f].mean()) for f in np.unique(test['focal'])}))
 print('evaluated',x['name'],flush=True)
m.js(O/'metrics.json',records);m.js(O/'training.json',timing)
# Average losses, never ensemble predictions. Shared cluster resamples across all contrasts.
rows=[]
for pct in P['fractions']:
 avg={c:np.mean([v.mean(0) for (q,r,k),v in losses.items() if q==pct and k==c],0) for c in P['conditions']+['ridge','rf','xgb','constant']}
 du=avg['LU']-avg['RU']
 contrasts=[('strategy','primary',4,avg['LF']-avg['RU']),('updated_source','secondary',20,du),('learned_update','secondary',20,avg['LU']-avg['LF'])]+[(c,'baseline',12,avg['LF']-avg[c]) for c in ['ridge','rf','xgb']]
 for name,family,size,d in contrasts:rows.append(dict(pct=pct,contrast=name,family=family,size=size,d=d))
clusters,inv=np.unique(test['cluster'],return_inverse=True);counts=np.bincount(inv);sums=np.array([np.bincount(inv,weights=x['d']) for x in rows]);rng=np.random.default_rng(P['bootstrap_seed']);draws=[]
for start in range(0,P['bootstrap_repeats'],200):
 ids=rng.integers(0,len(clusters),(min(200,P['bootstrap_repeats']-start),len(clusters)));weights=np.array([np.bincount(z,minlength=len(clusters)) for z in ids]);draws.append((weights@sums.T)/(weights@counts)[:,None])
bt=np.concatenate(draws);out=[]
for j,x in enumerate(rows):
 d=x.pop('d');x['difference']=float(d.mean());x['interval']=np.quantile(bt[:,j],[.025/x['size'],1-.025/x['size']]).tolist();out.append(x)
m.js(O/'contrasts.json',out)
(O/'README.md').write_text('# Training-data fraction evaluation\n\nmetrics.json: seed-level and focal-strain MSE for each subset and model. Prediction files: predictions for each evaluation row. training.json: selected steps, training and validation errors, and elapsed time. contrasts.json: paired differences by training fraction and family-adjusted intervals. Negative differences favor the first condition.\n\nIntervals use an affecting-strain-combination cluster bootstrap conditional on the completed training runs; they do not quantify generalization to new seeds, batches, or strains. Non-significance does not establish equivalence. The training-data reduction experiment holds 773 validation observations fixed. Repeats and seeds are not treated as 50 independent samples.\n')
print('EVALUATION COMPLETE',flush=True)
