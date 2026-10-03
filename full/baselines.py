"""Validation-only model selection. Never opens test files."""
from pathlib import Path
import sys,json,itertools,time
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from xgboost import XGBRegressor
R=Path(sys.argv[1]).resolve();P=json.loads((R/'plan.json').read_text());M=json.loads((R/'data/meta.json').read_text());O=R/'baselines';O.mkdir(exist_ok=True)
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def load(t,p):assert p in ['fit','val'];return dict(np.load(R/'data'/f'{t}_{p}.npz'))
def design(z,kind,meta):
 n=meta['n_entities'];ids=z['ids'];N=len(ids);aff=np.zeros((N,n))
 for j in range(ids.shape[1]-1):aff[np.arange(N),ids[:,j]]+=1
 foc=np.eye(n)[ids[:,-1]]
 if kind=='rf':return np.concatenate([aff,foc],axis=1).astype('float32')
 fs=sorted(set(meta['focal_mapping'].values()));fids=[meta['entities'].index(f) for f in fs];fmap={v:i for i,v in enumerate(fids)}
 rr,cc,vv=[],[],[]
 for r,ii in enumerate(ids):
  off=fmap[int(ii[-1])]*(n+1);rr.append(r);cc.append(off+n);vv.append(1.)
  for i in ii[:-1]:rr.append(r);cc.append(off+int(i));vv.append(1.)
 x=sparse.csr_matrix((vv,(rr,cc)),shape=(N,len(fids)*(n+1)))
 if kind=='pair':
  pairs={p:j for j,p in enumerate(itertools.combinations(range(n),2))};rr=[];cc=[]
  for r,ii in enumerate(ids):
   for a,b in itertools.combinations(sorted(ii[:-1]),2):rr.append(r);cc.append(pairs[(int(a),int(b))])
  x=sparse.hstack([x,sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(N,len(pairs)))],format='csr')
 return x

def extra_design(z,meta):
 base=design(z,'rf',meta);v=np.zeros((len(z['y']),meta['n_entities']),np.float32)
 for j in range(z['ids'].shape[1]-1):v[np.arange(len(v)),z['ids'][:,j]]=z['single'][:,j]
 return np.concatenate([base,v],axis=1)
def xgb_fit_select(xf,yf,xv,yv,out,primary=None):
 if primary is not None:settings=[json.loads(primary.read_text())['selected']]
 else:settings=[dict(max_depth=d,min_child_weight=c,reg_lambda=l) for d,c,l in itertools.product([2,4,6],[1,5],[1,10])]
 scores=[];best=float('inf')
 for j,setting in enumerate(settings):
  models=[];loss=[];iters=[]
  for seed in P['seeds']:
   cap=2000
   while True:
    q=XGBRegressor(objective='reg:squarederror',n_estimators=cap,learning_rate=.05,subsample=.8,colsample_bytree=1.,tree_method='hist',n_jobs=2,early_stopping_rounds=100,random_state=seed,**setting)
    q.fit(xf,yf,eval_set=[(xf,yf),(xv,yv)],verbose=False)
    n=len(q.evals_result()['validation_1']['rmse'])
    if n<cap or cap==16000:break
    cap*=2
   if n==16000 and q.best_iteration>=15900:raise RuntimeError('XGBoost budget incomplete; test remains closed')
   loss.append(float(np.mean((q.predict(xv)-yv)**2)));models.append(q);iters.append(int(q.best_iteration))
  score=float(np.mean(loss));scores.append(dict(setting=setting,mse=score,seed_mse=loss,best_iterations=iters))
  if score<best:best=score;selected=setting;joblib.dump(models,out.with_suffix('.joblib'),compress=3)
  save(out.parent/(out.name+'_progress.json'),dict(candidate=j+1,total=len(settings),best=best));print(R.name,out,'XGB',j+1,len(settings),score,flush=True)
 save(out.parent/(out.name+'_selection.json'),dict(selected=selected,candidates=scores))

def ridge_focal(fit,val,task):
 # Fit a separate unpenalized intercept for each focal strain.
 n=M['n_entities']
 def features(z):return design(z,'rf',M)[:,:n]
 xf,xv=features(fit),features(val);candidates=[]
 for a in P['ridge_alphas']:
  models={};pr=np.zeros(len(val['y']))
  for f in np.unique(fit['focal']):
   ix=fit['focal']==f;iv=val['focal']==f;q=Ridge(alpha=a,solver='svd').fit(xf[ix],fit['y'][ix]);models[str(f)]=q
   if iv.any():pr[iv]=q.predict(xv[iv])
  candidates.append((float(np.mean((pr-val['y'])**2)),a,models))
 best=min(candidates,key=lambda x:(x[0],x[1]));return best[2],dict(alpha=best[1],candidates=[dict(mse=a,alpha=b) for a,b,_ in candidates])

def run():
 for task in ['source','hour48','trio']:
  fit=load(task,'fit');val=load(task,'val');out=O/task;out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():continue
  q,info=ridge_focal(fit,val,task);joblib.dump(q,out/'ridge.joblib');save(out/'ridge_selection.json',info)
  xf=design(fit,'rf',M);xv=design(val,'rf',M)
  # The alternative split uses the RF configuration selected on the primary split.
  if R.name=='robust':
   info=json.loads((R.parent/'primary/baselines'/task/'rf_selection.json').read_text());settings=[info['selected']]
  else:settings=[dict(max_features=f,min_samples_leaf=l,max_depth=d) for f,l,d in itertools.product([1.,.5,'sqrt'],[1,3,10],[None,8])]
  best_loss=float('inf');scores=[]
  for j,setting in enumerate(settings):
   mods=[];loss=[]
   for seed in P['seeds']:
    q=RandomForestRegressor(n_estimators=500,random_state=seed,n_jobs=2,**setting).fit(xf,fit['y']);mods.append(q);loss.append(float(np.mean((q.predict(xv)-val['y'])**2)))
   score=float(np.mean(loss));scores.append(dict(setting=setting,mean_mse=score,seed_mse=loss))
   if score<best_loss:best_loss=score;best_setting=setting;joblib.dump(mods,out/'rf.joblib',compress=3)
   save(out/'progress.json',dict(candidate=j+1,total=len(settings),best=best_loss));print(R.name,task,'RF',j+1,len(settings),score,flush=True)
  save(out/'rf_selection.json',dict(selected=best_setting,candidates=scores))
  prior=R.parent/'primary/baselines'/task/'xgb_selection.json' if R.name=='robust' else None
  xgb_fit_select(xf,fit['y'],xv,val['y'],out/'xgb',prior)
  save(out/'complete.json',dict(finished=time.time()))
 # Linear reader: sum influencing latents, focal latent separately, permutation invariant.
 for task in ['hour48','trio']:
  fit=load(task,'fit');val=load(task,'val')
  for rep in ['gm']:
   for seed in P['seeds']:
    path=R/'runs'/f'reader_{task}_{rep}_{seed}'/'initial_latents.npz';u=np.load(path)['latent']
    def features(z):return np.concatenate([u[z['ids'][:,:-1]].sum(1),u[z['ids'][:,-1]]],axis=1)
    xf,xv=features(fit),features(val);cand=[]
    for a in P['ridge_alphas']:
     q=Ridge(alpha=a,solver='svd').fit(xf,fit['y']);cand.append((float(np.mean((q.predict(xv)-val['y'])**2)),a,q))
    best=min(cand,key=lambda t:t[0]);name=f'linear_{task}_{rep}_{seed}';joblib.dump(best[2],O/(name+'.joblib'));save(O/(name+'.json'),dict(alpha=best[1],val_mse=best[0],features='sum_affecting_latents_plus_focal'))
 if R.name=='primary':
  fit=load('source','fit');val=load('source','val');xf=extra_design(fit,M);xv=extra_design(val,M);best=float('inf');scores=[]
  for leaf,depth in itertools.product([1,3,10],[None,8]):
   models=[RandomForestRegressor(n_estimators=500,min_samples_leaf=leaf,max_depth=depth,random_state=seed,n_jobs=2).fit(xf,fit['y']) for seed in P['seeds']]
   mse=float(np.mean([(q.predict(xv)-val['y'])**2 for q in models]));scores.append(dict(leaf=leaf,depth=depth,mse=mse))
   if mse<best:best=mse;joblib.dump(models,O/'extra_rf.joblib',compress=3)
  save(O/'extra_rf_selection.json',dict(candidates=scores,best=best))
  xgb_fit_select(xf,fit['y'],xv,val['y'],O/'extra_xgb')
 save(O/'complete.json',dict(finished=time.time()))
if __name__=='__main__':run()
