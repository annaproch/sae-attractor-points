import json,csv
from pathlib import Path
import numpy as np
from scipy import sparse
p=Path('/workspace/project/runs/steering-pilot-001'); out=p/'analysis'
rows=json.loads((p/'summary.json').read_text())
states={}
for f in sorted((p/'states').glob('step_*.npz')):
 t=int(f.stem.split('_')[1]); ids=np.load(p/'states'/f'probe_ids_{t:05d}.npy'); a=sparse.load_npz(f).toarray()
 for i,z in zip(ids,a):states[int(i),t]=z.astype(float)
def jac(a,b):
 a=a!=0;b=b!=0
 return float(np.sum(a&b)/max(1,np.sum(a|b)))
def cosine(a,b):return float(a@b/max(1e-20,np.linalg.norm(a)*np.linalg.norm(b)))
metrics=[]
for r in rows:
 i=r['probe_id']; base=9*r['prompt_id']; z0=states[i,0]
 # Qualitative response review: repetition/symbol collapse, not factual correctness.
 deg='yes' if r['alpha']>=4 or (r['condition']=='random' and r['alpha']==2 and r['prompt_id'] in (1,3,4)) else 'uncertain' if r['condition']=='random' and r['alpha']==2 else 'no_clear_collapse'
 for t in range(6):
  z=states[i,t]; b=states[base,t]
  metrics.append(dict(probe_id=i,prompt_id=r['prompt_id'],condition=r['condition'],alpha=r['alpha'],review_collapse=deg,step=t,l0=int(np.count_nonzero(z)),norm_ratio=float(np.linalg.norm(z)/np.linalg.norm(z0)),baseline_jaccard=jac(z,b),baseline_cosine=cosine(z,b),initial_jaccard=jac(z,z0),turnover=0 if t==0 else 1-jac(z,states[i,t-1]),initial_error=r['relative_reconstruction_error'],final_step=r['final_step']))
with (out/'early_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
print('GROUP MEDIANS: condition alpha | l0 t0,1,3,5 | growth t5 | baseline J t0,t5 | turnover t1,t5')
for key in dict.fromkeys((r['condition'],r['alpha']) for r in rows):
 m=[r for r in metrics if (r['condition'],r['alpha'])==key]
 med=lambda field,t:round(float(np.median([r[field] for r in m if r['step']==t])),3)
 print(key,[med('l0',t) for t in (0,1,3,5)],med('norm_ratio',5),[med('baseline_jaccard',t) for t in (0,5)],[med('turnover',t) for t in (1,5)])
def auc(x,y):
 a=np.array(x)[np.array(y)==1]; b=np.array(x)[np.array(y)==0]
 return float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))
print('EXPLORATORY AUC excluding 2 uncertain cases')
for field,t,sign in [('initial_error',0,1),('l0',0,1),('norm_ratio',1,1),('norm_ratio',3,1),('norm_ratio',5,1),('turnover',1,1),('baseline_jaccard',0,-1),('baseline_jaccard',5,-1),('final_step',0,-1)]:
 m=[r for r in metrics if r['step']==t and r['review_collapse']!='uncertain']; print(field,t,round(auc([sign*r[field] for r in m],[r['review_collapse']=='yes' for r in m]),3))
print('EXAMPLES strongest IDs (amplitude) and l0/growth')
for i in [9,12,14,13,36,39,41]:
 print('PROBE',i)
 for t in [0,1,3,5]:
  z=states[i,t];top=np.argsort(z)[-5:][::-1];print(t, np.count_nonzero(z),round(np.linalg.norm(z)/np.linalg.norm(states[i,0]),3),[(int(j),round(float(z[j]),1)) for j in top])
print('common strongest at step5 by condition')
for key in dict.fromkeys((r['condition'],r['alpha']) for r in rows):
 ids=[r['probe_id'] for r in rows if (r['condition'],r['alpha'])==key]
 from collections import Counter
 print(key,Counter(int(j) for i in ids for j in np.argsort(states[i,5])[-10:]).most_common(5))
