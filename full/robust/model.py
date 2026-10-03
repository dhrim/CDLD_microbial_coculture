"""Asymmetric 2-to-1 CDLD with explicit selected-member value transfer.
Training APIs load fit/validation only; evaluation code opens test after selection.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('TF_DETERMINISTIC_OPS','1');os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
from pathlib import Path
import json,itertools
import numpy as np
import tensorflow as tf
R=Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text());M=json.loads((R/'data/meta.json').read_text());B=P['batch'];D=P['latent_dim']
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
def key(*v):return int(np.random.SeedSequence(v).generate_state(1)[0]%2147483647)
def js(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False,ensure_ascii=False));tmp.replace(path)
def load(task,part):
 assert part in ('fit','val'), 'Test is unavailable to training'
 return dict(np.load(R/'data'/f'{task}_{part}.npz'))
def initial(seed):return np.random.default_rng(key(1,seed)).normal(0,.1,(M['n_entities'],D)).astype('float32')
def optimizer(v):
 o=tf.keras.optimizers.Adam(P['lr']);o.build(v);return o
def reset(o):
 for v in o.variables:
  if 'learning_rate' not in v.name:v.assign(tf.zeros_like(v))
def penalty(n):return tf.add_n([tf.reduce_sum(v*v) for v in n.trainable_variables if 'bias' not in v.name])
class Net(tf.Module):
 def __init__(self,k,flags,seed,name):
  super().__init__(name=name);self.k=k;self.flags=flags
  tf.keras.utils.set_random_seed(seed)
  self.input_drop=[tf.keras.layers.Dropout(P['dropout'],seed=key(seed,2,j)) for j in range(k)]
  self.hidden_drop=tf.keras.layers.Dropout(P['dropout'],seed=key(seed,3))
  self.mlp=tf.keras.Sequential([tf.keras.Input((k*(D+2*flags),)),tf.keras.layers.Dense(64,activation='relu'),self.hidden_drop,tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1)])
  # Keras 3 random generators are explicit checkpoint dependencies.
  self.rngs=[d.seed_generator.state.value for d in self.input_drop+[self.hidden_drop]]
 @property
 def variables_to_train(self):return self.mlp.trainable_variables
 def __call__(self,z,training=False):
  z=tf.stack([d(z[:,j],training=training) for j,d in enumerate(self.input_drop)],axis=1)
  if self.flags:
   flag=tf.constant([[1.,0.]]*(self.k-1)+[[0.,1.]])
   z=tf.concat([z,tf.broadcast_to(flag,[tf.shape(z)[0],self.k,2])],axis=-1)
  return self.mlp(tf.reshape(z,[-1,self.k*(D+2*self.flags)]),training=training)[:,0]
def shuffled(ids,seed):
 n=tf.shape(ids)[0];k=ids.shape[1]-1
 order=tf.argsort(tf.random.stateless_uniform([n,k],[seed,1]),axis=1)
 return tf.concat([tf.gather(ids[:,:k],order,batch_dims=1),ids[:,k:]],axis=1)
def sym_predict(net,table,ids):
 vals=[];k=ids.shape[1]-1
 for perm in itertools.permutations(range(k)):
  order=list(perm)+[k];v=[]
  for st in range(0,len(ids),1024):v.append(net(tf.gather(table,tf.constant(ids[st:st+1024,order])),False).numpy())
  vals.append(np.concatenate(v))
 return np.mean(vals,axis=0)
class Finder:
 def __init__(self,seed,mode):
  self.mode=mode;self.u0=initial(seed);self.table=tf.Variable(self.u0,trainable=(mode=='joint'),name='latent_table');self.selected=tf.Variable(self.u0[0],name='selected')
  self.nets=[Net(3,True,key(10,seed,s),f'Finder_{s}') for s in range(2)]
  self.opts=[optimizer(n.variables_to_train) for n in self.nets]
  self.lo=optimizer([self.selected] if mode=='gm' else [self.table])
  self.ck=tf.train.Checkpoint(table=self.table,selected=self.selected,net0=self.nets[0],net1=self.nets[1],opt0=self.opts[0],opt1=self.opts[1],latent_opt=self.lo)
  self.visits=[self.make_visit(s) for s in range(2)]
 def make_visit(self,side):
  net=self.nets[side];opt=self.opts[side];gm=self.mode=='gm'
  @tf.function(input_signature=[tf.TensorSpec([None,3],tf.int32),tf.TensorSpec([None],tf.float32),tf.TensorSpec([],tf.int32),tf.TensorSpec([None],tf.int32)])
  def run(ids,y,player,keys):
   if gm:self.selected.assign(tf.gather(self.table,player));reset(self.lo)
   n=tf.shape(ids)[0];tr=tf.TensorArray(tf.float32,size=tf.shape(keys)[0])
   for j in tf.range(tf.shape(keys)[0]):
    sl=slice(j*B,tf.minimum((j+1)*B,n));ii=shuffled(ids[sl],keys[j])
    with tf.GradientTape() as tape:
     z=tf.gather(self.table,ii)
     if gm:z=tf.where((ii==player)[...,None],self.selected,tf.stop_gradient(z))
     pred=net(z,True);data=tf.reduce_mean((pred-y[sl])**2);loss=data+P['l2']*penalty(net.mlp)
    lv=self.selected if gm else self.table;vs=[lv]+net.variables_to_train;gs=tape.gradient(loss,vs)
    tf.debugging.assert_all_finite(loss,'nonfinite loss');self.lo.apply_gradients([(gs[0],lv)]);opt.apply_gradients(zip(gs[1:],vs[1:]))
    if gm:self.table.assign(tf.tensor_scatter_nd_update(self.table,[[player]],self.selected[None]))
    tr=tr.write(j,tf.stack([loss,data]))
   return tr.stack()
  return run
 def predict(self,z):return np.stack([sym_predict(n,self.table,z['ids']) for n in self.nets],axis=1)
class Reader:
 def __init__(self,latent,k,seed):
  self.table=tf.Variable(latent,trainable=False);self.net=Net(k,False,key(500,seed,k),'Predictor');self.opt=optimizer(self.net.variables_to_train)
  self.ck=tf.train.Checkpoint(table=self.table,net=self.net,opt=self.opt);self.epoch=self.make_epoch(k)
 def make_epoch(self,k):
  @tf.function(input_signature=[tf.TensorSpec([None,k],tf.int32),tf.TensorSpec([None],tf.float32),tf.TensorSpec([None],tf.int32)])
  def run(ids,y,keys):
   tr=tf.TensorArray(tf.float32,size=tf.shape(keys)[0]);n=tf.shape(ids)[0]
   for j in tf.range(tf.shape(keys)[0]):
    sl=slice(j*B,tf.minimum(n,(j+1)*B));ii=shuffled(ids[sl],keys[j])
    with tf.GradientTape() as tape:
     pred=self.net(tf.gather(self.table,ii),True);data=tf.reduce_mean((pred-y[sl])**2);loss=data+P['l2']*penalty(self.net.mlp)
    gs=tape.gradient(loss,self.net.variables_to_train);tf.debugging.assert_all_finite(loss,'nonfinite reader');self.opt.apply_gradients(zip(gs,self.net.variables_to_train));tr=tr.write(j,tf.stack([loss,data]))
   return tr.stack()
  return run
 def predict(self,z):return sym_predict(self.net,self.table,z['ids'])

class AugReader:
 """Frozen latent plus the identical measured-single information used by strongest."""
 def __init__(self,latent,k,seed):
  self.table=tf.Variable(latent,trainable=False);self.k=k
  tf.keras.utils.set_random_seed(key(600,seed,k))
  self.net=Net(k,False,key(500,seed,k),'AugPredictor')
  self.net.mlp=tf.keras.Sequential([tf.keras.Input((k*D+k-1,)),tf.keras.layers.Dense(64,activation='relu'),self.net.hidden_drop,tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1)])
  self.opt=optimizer(self.net.variables_to_train);self.ck=tf.train.Checkpoint(table=self.table,net=self.net,opt=self.opt)
  self.epoch=self.make_epoch(k)
 def forward(self,ids,extra,training):
  z=tf.gather(self.table,ids)
  z=tf.stack([d(z[:,j],training=training) for j,d in enumerate(self.net.input_drop)],axis=1)
  return self.net.mlp(tf.concat([tf.reshape(z,[-1,self.k*D]),extra],axis=1),training=training)[:,0]
 def make_epoch(self,k):
  @tf.function(input_signature=[tf.TensorSpec([None,k],tf.int32),tf.TensorSpec([None],tf.float32),tf.TensorSpec([None],tf.int32),tf.TensorSpec([None,k-1],tf.float32)])
  def run(ids,y,keys,extra):
   tr=tf.TensorArray(tf.float32,size=tf.shape(keys)[0]);n=tf.shape(ids)[0]
   for j in tf.range(tf.shape(keys)[0]):
    sl=slice(j*B,tf.minimum(n,(j+1)*B));ii=ids[sl];ee=extra[sl]
    order=tf.argsort(tf.random.stateless_uniform([tf.shape(ii)[0],k-1],[keys[j],1]),axis=1)
    ii=tf.concat([tf.gather(ii[:,:-1],order,batch_dims=1),ii[:,-1:]],axis=1);ee=tf.gather(ee,order,batch_dims=1)
    with tf.GradientTape() as tape:
     p=self.forward(ii,ee,True);data=tf.reduce_mean((p-y[sl])**2);loss=data+P['l2']*penalty(self.net.mlp)
    gs=tape.gradient(loss,self.net.variables_to_train);self.opt.apply_gradients(zip(gs,self.net.variables_to_train));tr=tr.write(j,tf.stack([loss,data]))
   return tr.stack()
  return run
 def predict(self,z):
  vals=[]
  for perm in itertools.permutations(range(self.k-1)):
   order=list(perm)+[self.k-1];v=[]
   for st in range(0,len(z['ids']),1024):
    v.append(self.forward(z['ids'][st:st+1024,order],z['single_std'][st:st+1024,list(perm)],False).numpy())
   vals.append(np.concatenate(v))
  return np.mean(vals,axis=0)
