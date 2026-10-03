import argparse,json,time,os,fcntl,traceback
from pathlib import Path
import numpy as np
import model as m
R=m.R;P=m.P;tf=m.tf

def train(stage,rep,seed,target):
 task='source' if stage in ('source','aug') else target
 fit=m.load(task,'fit');val=m.load(task,'val');sc=m.M['scales'][task];mu=sc['mean'];sd=sc['std'];y=((fit['y']-mu)/sd).astype('float32')
 out=R/'runs'/f'{stage}_{target}_{rep}_{seed}';out.mkdir(parents=True,exist_ok=True)
 lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'complete.json').exists():return
 if stage=='source':obj=m.Finder(seed,rep);config=P['finder']
 else:
  source=R/'runs'/f'source_source_{rep}_{seed}'
  latent=m.initial(seed) if rep=='initial' else np.load(source/'best_latents.npz')['latent']
  obj=(m.AugReader if stage=='aug' else m.Reader)(latent,fit['ids'].shape[1],seed);config=P['reader']
  if stage=='aug':
   em=float(fit['single'].mean());es=float(fit['single'].std());assert es>0
   for z in [fit,val]:z['single_std']=((z['single']-em)/es).astype('float32')
   m.js(out/'extra_scale.json',dict(mean=em,std=es))
 np.savez_compressed(out/'initial_latents.npz',latent=obj.table.numpy())
 nets=obj.nets if stage=='source' else [obj.net]
 np.savez_compressed(out/'initial_network_weights.npz',**{f'n{s}_w{j}':v.numpy() for s,n in enumerate(nets) for j,v in enumerate(n.variables_to_train)})
 def record(step,side=None):
  result={'step':step,'side':side}
  for name,z in [('fit',fit),('val',val)]:
   yy=(z['y']-mu)/sd;pred=obj.predict(z)
   if stage=='source':loss=((pred-yy[:,None])**2).mean(axis=0);result[name+'_sides']=loss.tolist();result[name]=float(loss.mean())
   else:result[name]=float(np.mean((pred-yy)**2))
  return result
 manager=tf.train.CheckpointManager(obj.ck,str(out/'resume_state'),max_to_keep=2)
 if (out/'resume.json').exists():
  state=json.loads((out/'resume.json').read_text());obj.ck.restore(str(out / state['checkpoint'])).expect_partial();start=state['step'];anchor=state['anchor'];last=state['last'];best=json.loads((out/'best.json').read_text())['val']
 else:
  start=0;rec=record(0);best=anchor=rec['val'];last=0;m.js(out/'step_0000.json',rec);m.js(out/'best.json',rec);obj.ck.write(str(out/'best_state'));np.savez_compressed(out/'best_latents.npz',latent=obj.table.numpy())
 pools=[[np.flatnonzero((fit['ids'][:,:2]==i).any(axis=1)) for i in range(m.M['n_entities'])],[np.flatnonzero(fit['ids'][:,2]==i) for i in range(m.M['n_entities'])]] if stage=='source' else None
 began=time.time();reason='cap';total_updates=0
 cap=config['cap']
 for step in range(start+1,config['max_cap']+1):
  tick=time.time();losses=[];presentations=0
  if stage=='source':
   order=np.random.default_rng(m.key(11,seed,step)).permutation(m.M['n_entities'])
   for side in range(2):
    members=order if rep=='gm' else [-1]
    for i in members:
     rows=pools[side][i] if rep=='gm' else np.arange(len(y)); rows=np.random.default_rng(m.key(12,seed,step,side,int(i)+1)).permutation(rows)
     if not len(rows):continue
     keys=np.array([m.key(13,seed,step,side,int(i)+1,j) for j in range((len(rows)+m.B-1)//m.B)],np.int32)
     tr=obj.visits[side](fit['ids'][rows],y[rows],np.int32(max(i,0)),keys).numpy();losses.extend(tr.tolist());presentations+=len(rows)
    phase=record(step,side);m.js(out/f'phase_{step:04d}_{side}.json',phase)
   rec=phase
  else:
   rows=np.random.default_rng(m.key(14,seed,step)).permutation(len(y));keys=np.array([m.key(15,seed,step,j) for j in range((len(rows)+m.B-1)//m.B)],np.int32)
   args=[fit['ids'][rows],y[rows],keys]
   if stage=='aug':args.append(fit['single_std'][rows])
   losses=obj.epoch(*args).numpy().tolist();presentations=len(y);rec=record(step)
  total_updates+=len(losses);rec.update(seconds=time.time()-tick,presentations=presentations,updates=len(losses),training_objective=float(np.mean(np.array(losses)[:,0])),training_data_loss=float(np.mean(np.array(losses)[:,1])))
  m.js(out/f'step_{step:04d}.json',rec)
  if rec['val']<best:
   best=rec['val'];m.js(out/'best.json',rec);obj.ck.write(str(out/'best_state'));np.savez_compressed(out/'best_latents.npz',latent=obj.table.numpy())
  if anchor-rec['val']>config['min_delta']:anchor=rec['val'];last=step
  ck=manager.save(checkpoint_number=step);m.js(out/'resume.json',dict(step=step,anchor=anchor,last=last,checkpoint=str(Path(ck).relative_to(out))))
  m.js(out/'progress.json',dict(pid=os.getpid(),step=step,best=best,elapsed=time.time()-began,**{k:v for k,v in rec.items() if k!='step'}))
  if step==1 or step%10==0:print(f'{stage}/{task}/{rep}/{seed} step {step} fit {rec["fit"]:.6f} val {rec["val"]:.6f} best {best:.6f} {rec["seconds"]:.2f}s',flush=True)
  if step>=config['min'] and step-last>=config['patience']:reason='patience';break
  if step>=cap:
   if cap>=config['max_cap']:break
   cap=min(cap*2,config['max_cap']);m.js(out/f'budget_extension_{step}.json',dict(cap=cap,last_improvement=last,reason='validation improvement at cap; no test access'))
 obj.ck.restore(str(out/'best_state')).expect_partial();pred=obj.predict(val)*sd+mu
 np.savez_compressed(out/'validation_predictions.npz',prediction=pred,y=val['y'],row_id=val['row_id'],cluster=val['cluster'])
 m.js(out/'complete.json',dict(stage=stage,representation=rep,target=task,seed=seed,steps=step,stop_reason=reason,budget_unresolved=(reason=='cap' and step-last<config['patience']),selected=json.loads((out/'best.json').read_text()),seconds=time.time()-began,updates=total_updates,finished=time.time()))
 print('COMPLETE',out.name,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['source','reader','aug']);a.add_argument('rep');a.add_argument('seed',type=int);a.add_argument('target');v=a.parse_args()
 try:train(v.stage,v.rep,v.seed,v.target)
 except BaseException:
  out=R/'runs'/f'{v.stage}_{v.target}_{v.rep}_{v.seed}';out.mkdir(parents=True,exist_ok=True);(out/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc());raise
