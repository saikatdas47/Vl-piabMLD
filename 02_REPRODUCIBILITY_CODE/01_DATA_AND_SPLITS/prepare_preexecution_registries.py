#!/usr/bin/env python3
"""Create real-cohort registries without fitting or scoring any model."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiment_system import hierarchical_subsets
from benchmark_pipeline import stable_seed
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REG=ROOT/'data/00_registry';SPL=ROOT/'data/04_splits';PLAN=ROOT/'data/06_preexecution';PLAN.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'config/experiment_config.json').read_text());freeze=json.loads((REG/'cohort_freeze_01.json').read_text())
# Registry generation never fits or scores a model and is safe regardless of execution authorization.
models=cfg['model_families']; rows=[]; sample=[]; datasets=[]
for d in freeze['datasets']:
 did=d['dataset_id'];f=pd.read_csv(ROOT/d['analysis_reference'],low_memory=False);y=f.__target__.astype(int);g=f.__group_id__.astype(str)
 tr,te=next(StratifiedGroupKFold(5,shuffle=True,random_state=stable_seed(cfg['master_seed'],'C1',did)).split(f,y,g))
 a=pd.DataFrame({'dataset_id':did,'source_row':f.__source_row__.astype(int),'group_id':g,'partition':'development'})
 a.loc[te,'partition']='test';p=SPL/f'{did}_C1_assignments.csv.gz';a.to_csv(p,index=False,compression='gzip')
 c=y.value_counts();datasets.append({'dataset_id':did,'rows':len(f),'predictors':len(f.columns)-3,'groups':g.nunique(),'negative':int(c.get(0,0)),'positive':int(c.get(1,0)),'minority':int(c.min()),'C1_development_rows':len(tr),'C1_test_rows':len(te),'C1_split_file':str(p.relative_to(ROOT))})
 sr=hierarchical_subsets(f,cfg['sample_fractions'],cfg['subsample_chains'],did,cfg['minimum_total_rows'],cfg['minimum_minority_rows']);sample.append(sr)
 membership=[];group_frame=f.groupby('__group_id__')['__target__'].agg(['count','sum']).reset_index()
 for chain in range(cfg['subsample_chains']):
  seed=stable_seed(cfg['master_seed'],'subsample',did,chain);order=group_frame.sample(frac=1,random_state=seed).__group_id__.astype(str).tolist();rank={v:i for i,v in enumerate(order)};ng=len(order)
  cut50=max(1,int(np.ceil(ng*.50)));cut75=max(1,int(np.ceil(ng*.75)))
  for source_row,group_id in zip(f.__source_row__.astype(int),g):
   r=rank[str(group_id)];entry=.5 if r<cut50 else .75 if r<cut75 else 1.0
   membership.append({'dataset_id':did,'chain':chain,'source_row':source_row,'group_id':group_id,'entry_fraction':entry,'seed':seed})
 pd.DataFrame(membership).to_csv(SPL/f'{did}_sample_chain_membership.csv.gz',index=False,compression='gzip')
 # Freeze every estimable sample-size outer assignment before any model is run.
 sample_outer_path=SPL/f'{did}_sample_outer_assignments.csv.gz';first_sample_outer=True
 for _,ss in sr[(sr.fraction<1)&(sr.status=='ESTIMABLE')].iterrows():
  chain=int(ss.chain);fraction=float(ss.fraction);mem=pd.DataFrame(membership);keep=set(mem[(mem.chain==chain)&(mem.entry_fraction<=fraction)].source_row.astype(int));sub=f[f.__source_row__.astype(int).isin(keep)].reset_index(drop=True)
  for rep in range(cfg['outer_repeats']):
   splitter=StratifiedGroupKFold(cfg['outer_folds'],shuffle=True,random_state=stable_seed(cfg['master_seed'],'sample_outer',did,chain,fraction,rep))
   assignment=np.full(len(sub),-1,int)
   for fold,(_,test) in enumerate(splitter.split(sub,sub.__target__,sub.__group_id__)):assignment[test]=fold
   for fold in range(cfg['outer_folds']):
    test=np.flatnonzero(assignment==fold);train=np.flatnonzero(assignment!=fold)
    if len(np.unique(sub.__target__.to_numpy()[test]))<2:raise RuntimeError(f'{did} sample outer test class infeasible')
    inner=StratifiedGroupKFold(cfg['inner_folds'],shuffle=True,random_state=stable_seed(cfg['master_seed'],'sample_inner',did,chain,fraction,rep,fold))
    for itr,iva in inner.split(sub.iloc[train],sub.__target__.to_numpy()[train],sub.__group_id__.astype(str).to_numpy()[train]):
     if len(np.unique(sub.__target__.to_numpy()[train][itr]))<2 or len(np.unique(sub.__target__.to_numpy()[train][iva]))<2:raise RuntimeError(f'{did} sample inner class infeasible')
     if set(sub.__group_id__.astype(str).to_numpy()[train][itr]) & set(sub.__group_id__.astype(str).to_numpy()[train][iva]):raise RuntimeError(f'{did} sample inner group leakage')
   chunk=pd.DataFrame({'dataset_id':did,'chain':chain,'fraction':fraction,'repeat':rep,'source_row':sub.__source_row__.astype(int),'group_id':sub.__group_id__.astype(str),'target':sub.__target__.astype(int),'outer_fold':assignment})
   chunk.to_csv(sample_outer_path,index=False,compression='gzip',mode='wt' if first_sample_outer else 'at',header=first_sample_outer);first_sample_outer=False
 # Full-cohort primary job plan. It is a registry, not execution authorization.
 for model in models:
  for cond in ['C0','C3','C4','C5','C6']:
   for rep in range(cfg['outer_repeats']):
    for fold in range(cfg['outer_folds']):rows.append({'dataset_id':did,'sample_fraction':1.0,'chain':'PRIMARY','condition':cond,'model':model,'repeat':rep,'fold':fold,'status':'PLANNED_LOCKED','authorization':'BLOCKED'})
  for rep in range(cfg['outer_repeats']):rows.append({'dataset_id':did,'sample_fraction':1.0,'chain':'PRIMARY','condition':'C2','model':model,'repeat':rep,'fold':'NON_NESTED','status':'PLANNED_LOCKED','authorization':'BLOCKED'})
  rows.append({'dataset_id':did,'sample_fraction':1.0,'chain':'PRIMARY','condition':'C1','model':model,'repeat':0,'fold':'FIXED_80_20','status':'PLANNED_LOCKED','authorization':'BLOCKED'})
sample=pd.concat(sample,ignore_index=True);sample.to_csv(PLAN/'sample_size_registry.csv',index=False);pd.DataFrame(datasets).to_csv(PLAN/'dataset_compute_inputs.csv',index=False);primary=pd.DataFrame(rows);primary.to_csv(PLAN/'primary_job_registry.csv',index=False)
# Secondary sample-size jobs only for estimable lower fractions. C0 is paired with C2-C5; C1/C6 are controls and excluded.
sec=[]
for _,s in sample[(sample.fraction<1)&(sample.status=='ESTIMABLE')].iterrows():
 for model in models:
  for cond in ['C0','C3','C4','C5']:
   for rep in range(cfg['outer_repeats']):
    for fold in range(cfg['outer_folds']):sec.append({'dataset_id':s.dataset_id,'sample_fraction':s.fraction,'chain':int(s.chain),'condition':cond,'model':model,'repeat':rep,'fold':fold,'status':'PLANNED_LOCKED','authorization':'BLOCKED'})
  for rep in range(cfg['outer_repeats']):sec.append({'dataset_id':s.dataset_id,'sample_fraction':s.fraction,'chain':int(s.chain),'condition':'C2','model':model,'repeat':rep,'fold':'NON_NESTED','status':'PLANNED_LOCKED','authorization':'BLOCKED'})
secondary=pd.DataFrame(sec);secondary.to_csv(PLAN/'sample_size_job_registry.csv',index=False)
summary={'registry_version':'PREEXEC-01','registry_is_execution_authority':False,'runtime_authorization_source':'config/REAL_EXPERIMENT_AUTHORIZATION.json; evaluated separately at launch','dataset_count':len(datasets),'primary_jobs':len(primary),'estimable_lower_fraction_dataset_chains':int(len(sample[(sample.fraction<1)&(sample.status=='ESTIMABLE')])),'sample_size_jobs':len(secondary),'total_planned_jobs':len(primary)+len(secondary),'job_definition':'one immutable summary and prediction archive unit','config_sha256':hashlib.sha256((ROOT/'config/experiment_config.json').read_bytes()).hexdigest()}
(PLAN/'compute_plan_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
