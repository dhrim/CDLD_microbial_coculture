"""Reduced-data training with Cycle Dual discovery and a frozen-latent Predictor."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'full/primary'
sys.path.insert(0,str(OLD))
import model as m
import run_job as runner
subset,condition,seed,stage=sys.argv[1:];seed=int(seed)
assert condition in ['LF','RF0','LU','RU']; assert stage in ['source','reader']
assert stage!='source' or condition in ['LU','RU']
base=ROOT/'subsets'/subset;out=base/condition
out.mkdir(exist_ok=True)
m.M=json.loads((base/'data/meta.json').read_text());m.P=json.loads((ROOT/'plan.json').read_text());runner.P=m.P
m.R=runner.R=out
initial_path=OLD/'runs'/f'reader_hour48_initial_{seed}'/'initial_latents.npz'
learned_path=OLD/'runs'/f'source_source_gm_{seed}'/'best_latents.npz'
u=np.load(learned_path)['latent'] if condition.startswith('L') else m.initial(seed)
if stage=='reader' and condition.endswith('U'):
 discovery=out/'runs'/f'source_hour48_gm_{seed}'
 c=json.loads((discovery/'complete.json').read_text());assert not c['budget_unresolved']
 u=np.load(discovery/'best_latents.npz')['latent']
m.initial=lambda s:u.copy()
def load(task,part):
 assert part in ('fit','val')
 return dict(np.load(base/'data'/f'hour48_{part}.npz'))
m.load=load
runner.train(stage,'gm' if stage=='source' else 'initial',seed,'hour48')
