#!/usr/bin/env python3
"""Shared resumable runner for one frozen dataset and selectable model families.

Real mode is disabled unless both the configuration gate and a separate release
authorization file agree. Plan and synthetic modes never fit biomedical data.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,os,resource,shutil,sys,time,traceback,warnings
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiment_system import execute_outer,sha,utc
from benchmark_pipeline import stable_seed
import numpy as np,pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold

CONFIG=ROOT/'config/experiment_config.json';FREEZE=ROOT/'data/00_registry/cohort_freeze_01.json';RESULTS=ROOT/'results';TRACK=ROOT/'data/07_execution_tracking';TRACK.mkdir(parents=True,exist_ok=True)

def append_jsonl(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a',encoding='utf-8') as f:f.write(json.dumps(obj,sort_keys=True,default=str)+'\n')
def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n');os.replace(tmp,path)
def canonical_hash(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def safe_name(v):return str(v).replace('.','p').replace('/','_')
def storage_snapshot(cfg,mode):
 u=shutil.disk_usage(ROOT);free=u.free/(1024**3);policy=cfg['storage_policy'];drive_fuse=str(ROOT).startswith('/content/drive/')
 reliable=not drive_fuse
 status='CAUTION' if drive_fuse else 'STOP' if free<float(policy['minimum_free_gib_before_new_atomic_run']) else 'CAUTION' if free<float(policy['warning_free_gib']) else 'PASS'
 row={'utc':utc(),'mode':mode,'status':status,'free_gib':round(free,3),'used_gib':round(u.used/(1024**3),3),'capacity_reading_reliable':reliable,'capacity_note':'Google Drive FUSE quota values are recorded but never used to block a run; actual write failures remain explicit and resumable.' if drive_fuse else None,'results_bytes':sum(p.stat().st_size for p in RESULTS.rglob('*') if p.is_file()) if RESULTS.exists() else 0};append_jsonl(TRACK/'storage_snapshots.jsonl',row);return row

def peak_rss_bytes():
 """Return peak resident-set size in bytes on supported Unix platforms."""
 raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return int(raw if sys.platform=='darwin' else raw*1024)

class ProductionStore:
 def __init__(self,root):self.root=Path(root)
 def paths(self,job):
  d=self.root/job['dataset_id']/job['model']/job['phase']/f"fraction_{safe_name(job['sample_fraction'])}"/f"chain_{safe_name(job['chain'])}"/job['condition']/f"repeat_{int(job['repeat']):02d}"/f"fold_{safe_name(job['fold'])}"/job['run_id']
  return d,d/'summary.json',d/'predictions.csv.gz',d/'provenance.json.gz',d/'manifest.json'
 def complete(self,job,expected):
  d,s,p,v,m=self.paths(job)
  if not all(x.exists() for x in [s,p,v,m]):return False
  z=json.loads(m.read_text())
  return z.get('status')=='COMPLETE' and all(z.get(k)==val for k,val in expected.items()) and sha(s)==z.get('summary_sha256') and sha(p)==z.get('predictions_sha256') and sha(v)==z.get('provenance_sha256')
 def save(self,job,summary,preds,provenance,manifest):
  d,s,p,v,m=self.paths(job);d.mkdir(parents=True,exist_ok=True)
  if any(x.exists() for x in [s,p,v,m]):raise RuntimeError('partial or mismatched run exists; refusing overwrite: '+str(d))
  st=s.with_suffix('.json.tmp');pt=p.with_suffix('.gz.tmp');vt=v.with_suffix('.gz.tmp');mt=m.with_suffix('.json.tmp')
  st.write_text(json.dumps(summary,indent=2,default=str)+'\n');preds.to_csv(pt,index=False,compression='gzip')
  with gzip.open(vt,'wt',encoding='utf-8') as f:json.dump(provenance,f,sort_keys=True,default=str)
  manifest|={'status':'COMPLETE','summary_sha256':sha(st),'predictions_sha256':sha(pt),'provenance_sha256':sha(vt),'completed_utc':utc()};mt.write_text(json.dumps(manifest,indent=2,default=str)+'\n')
  os.replace(st,s);os.replace(pt,p);os.replace(vt,v);os.replace(mt,m)

def synthetic_frame(dataset_id):
 seed=stable_seed(20260910,'runner-synthetic',dataset_id);X,y=make_classification(n_samples=260,n_features=9,n_informative=5,n_redundant=2,weights=[.68,.32],random_state=seed)
 f=pd.DataFrame(X,columns=[f'x{i}' for i in range(X.shape[1])]);f['category']=np.where(f.x0>0,'A','B');f.loc[f.index%17==0,'x2']=np.nan;f['__source_row__']=np.arange(len(f));f['__group_id__']=[f'SYN_{i:04d}' for i in range(len(f))];f['__target__']=y;return f
def load_real(dataset_id):
 freeze=json.loads(FREEZE.read_text());entry=next((x for x in freeze['datasets'] if x['dataset_id']==dataset_id),None)
 if not entry:raise KeyError(dataset_id)
 return pd.read_csv(ROOT/entry['analysis_reference'],low_memory=False),entry
def authorize_real(cfg):
 p=ROOT/'config/REAL_EXPERIMENT_AUTHORIZATION.json'
 if not p.exists():raise RuntimeError('REAL EXECUTION BLOCKED: a signed authorization file is required')
 a=json.loads(p.read_text())
 if a.get('authorized') is not True or a.get('release_id')!=cfg.get('active_release_id'):raise RuntimeError('REAL EXECUTION BLOCKED: authorization does not match active release')
 if cfg.get('main_experiment_enabled') is not True and a.get('authorization_scope')!='ONE_BOUNDED_SMOKE_UNIT_ONLY':raise RuntimeError('REAL EXECUTION BLOCKED: global gate is closed and authorization is not a bounded smoke unit')
 return a

def enforce_bounded_scope(args,dataset_id,jobs,authorization):
 if authorization.get('authorization_scope')!='ONE_BOUNDED_SMOKE_UNIT_ONLY':return
 if args.max_units!=1 or len(jobs)!=1:raise RuntimeError('REAL EXECUTION BLOCKED: bounded smoke authorization requires exactly one unit')
 job=jobs[0]
 expected={'dataset_id':dataset_id,'model':job['model'],'condition':job['condition'],'phase':job['phase'],'repeat':int(job['repeat']),'fold':int(job['fold']),'max_units':1}
 actual={k:authorization.get(k) for k in expected}
 actual['repeat']=int(actual['repeat']) if actual['repeat'] is not None else None
 actual['fold']=int(actual['fold']) if actual['fold'] is not None else None
 if actual!=expected:raise RuntimeError('REAL EXECUTION BLOCKED: requested unit is outside the bounded authorization scope')
def subset_frame(frame,dataset_id,chain,fraction):
 if float(fraction)==1:return frame.reset_index(drop=True)
 mem=pd.read_csv(ROOT/f'data/04_splits/{dataset_id}_sample_chain_membership.csv.gz');keep=set(mem[(mem.chain==int(chain))&(mem.entry_fraction<=float(fraction))].source_row.astype(int));return frame[frame.__source_row__.astype(int).isin(keep)].reset_index(drop=True)
def outer_indices(frame,dataset_id,repeat,fold,mode,chain,fraction):
 if mode=='synthetic':
  n=json.loads(CONFIG.read_text())['outer_folds'];splits=list(StratifiedGroupKFold(n,shuffle=True,random_state=stable_seed(20260910,'synouter',dataset_id,chain,fraction,repeat)).split(frame,frame.__target__,frame.__group_id__));return splits[int(fold)]
 if float(fraction)==1:
  a=pd.read_csv(ROOT/f'data/04_splits/{dataset_id}_outer_assignments.csv.gz');test_rows=set(a[(a.repeat==int(repeat))&(a.outer_fold==int(fold))].source_row.astype(int));test=np.flatnonzero(frame.__source_row__.astype(int).isin(test_rows));train=np.setdiff1d(np.arange(len(frame)),test);return train,test
 a=pd.read_csv(ROOT/f'data/04_splits/{dataset_id}_sample_outer_assignments.csv.gz');q=a[(a.chain==int(chain))&(np.isclose(a.fraction.astype(float),float(fraction)))&(a.repeat==int(repeat))];test_rows=set(q[q.outer_fold==int(fold)].source_row.astype(int));test=np.flatnonzero(frame.__source_row__.astype(int).isin(test_rows));train=np.setdiff1d(np.arange(len(frame)),test);return train,test
def source_fold_vector(frame,dataset_id,repeat,mode,chain,fraction):
 out=np.full(len(frame),-1,int)
 for fold in range(json.loads(CONFIG.read_text())['outer_folds']):
  _,te=outer_indices(frame,dataset_id,repeat,fold,mode,chain,fraction);out[te]=fold
 return out
def c1_indices(frame,dataset_id,mode):
 if mode=='synthetic':return next(StratifiedGroupKFold(5,shuffle=True,random_state=stable_seed(20260910,'C1',dataset_id)).split(frame,frame.__target__,frame.__group_id__))
 a=pd.read_csv(ROOT/f'data/04_splits/{dataset_id}_C1_assignments.csv.gz');test_rows=set(a[a.partition=='test'].source_row.astype(int));te=np.flatnonzero(frame.__source_row__.astype(int).isin(test_rows));return np.setdiff1d(np.arange(len(frame)),te),te

def planned_jobs(dataset_id,models,phase,conditions,mode,max_units):
 if mode=='synthetic':
  rows=[{'dataset_id':dataset_id,'sample_fraction':1.0,'chain':'SYNTHETIC','condition':c,'model':m,'repeat':0,'fold':('NON_NESTED' if c=='C2' else 'FIXED_80_20' if c=='C1' else 0),'phase':'synthetic'} for m in models for c in conditions]
 else:
  parts=[]
  if phase in {'primary','all'}:parts.append(pd.read_csv(ROOT/'data/06_preexecution/primary_job_registry.csv'))
  if phase in {'sample','all'}:parts.append(pd.read_csv(ROOT/'data/06_preexecution/sample_size_job_registry.csv'))
  q=pd.concat(parts,ignore_index=True);q=q[(q.dataset_id==dataset_id)&q.model.isin(models)&q.condition.isin(conditions)];rows=q.to_dict('records')
  for x in rows:x['phase']='primary' if float(x['sample_fraction'])==1 else 'sample_size'
 if max_units is not None:rows=rows[:max_units]
 return rows
def make_id(job,cfg_hash,code_hash,engine_hash,pipeline_hash):
 base={k:job[k] for k in ['dataset_id','model','condition','phase','sample_fraction','chain','repeat','fold']};base|={'config_sha256':cfg_hash,'runner_sha256':code_hash,'engine_sha256':engine_hash,'pipeline_sha256':pipeline_hash};return '__'.join([safe_name(base[k]) for k in ['dataset_id','model','phase','sample_fraction','chain','condition','repeat','fold']])+'__'+canonical_hash(base)[:16]

def run(dataset_id,argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='all');ap.add_argument('--mode',choices=['plan','synthetic','real'],default='plan');ap.add_argument('--phase',choices=['primary','sample','all'],default='primary');ap.add_argument('--conditions',default='C0,C1,C2,C3,C4,C5,C6');ap.add_argument('--max-units',type=int);ap.add_argument('--n-iter',type=int);ap.add_argument('--continue-on-error',action='store_true');ap.add_argument('--no-infrastructure-retry',action='store_true');args=ap.parse_args(argv)
 cfg=json.loads(CONFIG.read_text());models=cfg['model_families'] if args.model=='all' else [args.model]
 if any(m not in cfg['model_families'] for m in models):raise SystemExit('Unknown model')
 conditions=[x.strip() for x in args.conditions.split(',') if x.strip()]
 if any(c not in cfg['conditions'] for c in conditions):raise SystemExit('Unknown condition')
 authorization=None
 if args.mode=='real':
  authorization=authorize_real(cfg);space=storage_snapshot(cfg,args.mode)
  if space['status']=='STOP':raise RuntimeError('REAL EXECUTION BLOCKED: free-space reserve reached')
 print(f'[RUNNER] {dataset_id}: mode={args.mode}, models={", ".join(models)}, phase={args.phase}',flush=True)
 frame=synthetic_frame(dataset_id) if args.mode=='synthetic' else (None if args.mode=='plan' else load_real(dataset_id)[0])
 if frame is not None:print(f'[DATA] {dataset_id}: loaded once; rows={len(frame)}, columns={len(frame.columns)}',flush=True)
 jobs=planned_jobs(dataset_id,models,args.phase,conditions,args.mode,args.max_units);model_order={m:i for i,m in enumerate(models)};jobs.sort(key=lambda x:(model_order[x['model']],0 if x['phase']=='primary' else 1,float(x['sample_fraction']),str(x['chain']),x['condition'],int(x['repeat']),str(x['fold'])));cfg_hash=sha(CONFIG);code_hash=sha(Path(__file__));engine_hash=sha(ROOT/'code/experiment_system.py');pipeline_hash=sha(ROOT/'code/benchmark_pipeline.py');store=ProductionStore(RESULTS if args.mode=='real' else ROOT/'data/08_runner_synthetic/results')
 if args.mode=='real':enforce_bounded_scope(args,dataset_id,jobs,authorization)
 index=[];current_model=None;processed=0;total_jobs=len(jobs);sequence_start=time.perf_counter()
 print(f'[PLAN] {dataset_id}: {total_jobs} atomic evaluation unit(s)',flush=True)
 queue=[dict(x) for x in jobs]
 for job in queue:
  if job['model']!=current_model:
   if current_model is not None:print(f'\n[MODEL COMPLETE] {dataset_id} / {current_model}',flush=True)
   print(f'\n[MODEL START] {dataset_id} / {job["model"]}',flush=True)
  current_model=job['model']
  job['run_id']=make_id(job,cfg_hash,code_hash,engine_hash,pipeline_hash);expected={'config_sha256':cfg_hash,'runner_sha256':code_hash,'engine_sha256':engine_hash,'pipeline_sha256':pipeline_hash};d,s,p,v,m=store.paths(job)
  processed+=1
  label=f'{job["condition"]} {job["phase"]} fraction={job["sample_fraction"]} repeat={job["repeat"]} fold={job["fold"]}'
  if args.mode=='plan':index.append(job|{'status':'PLANNED','output_directory':str(d)});print(f'\r[PROGRESS] {processed}/{total_jobs} planned | {job["model"]} | {label}',end='',flush=True);continue
  attempt={'run_id':job['run_id'],'dataset_id':dataset_id,'model':job['model'],'event':'START_OR_RESUME','mode':args.mode,'utc':utc()};append_jsonl(TRACK/f'{args.mode}_attempt_log.jsonl',attempt)
  if store.complete(job,expected):
   index.append(job|{'status':'SKIPPED_COMPLETE','output_directory':str(d)});print(f'\r[PROGRESS] {processed}/{total_jobs} SKIPPED_COMPLETE | {job["model"]} | {label}',end='',flush=True)
   if processed%25==0 or processed==total_jobs:print(flush=True)
   continue
  print(f'\r[PROGRESS] {processed}/{total_jobs} RUNNING | {job["model"]} | {label}',end='',flush=True)
  start=time.perf_counter();rss0=peak_rss_bytes()
  try:
   work=subset_frame(frame,dataset_id,job['chain'],job['sample_fraction']) if args.mode=='real' else frame
   if job['condition']=='C1':tr,te=c1_indices(work,dataset_id,args.mode);fold_num=0;source_folds=None
   elif job['condition']=='C2':tr=np.arange(len(work));te=np.arange(len(work));fold_num=0;source_folds=None
   else:tr,te=outer_indices(work,dataset_id,int(job['repeat']),int(job['fold']),args.mode,job['chain'],job['sample_fraction']);fold_num=int(job['fold']);source_folds=source_fold_vector(work,dataset_id,int(job['repeat']),args.mode,job['chain'],job['sample_fraction']) if job['condition']=='C5' else None
   n_iter=args.n_iter if args.n_iter is not None else cfg['hyperparameter_candidates_per_model']
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');met,pred,prov=execute_outer(work,tr,te,job['condition'],job['model'],dataset_id,int(job['repeat']),fold_num,n_iter=n_iter,source_folds=source_folds)
   warning_counts=Counter((w.category.__name__,str(w.message)) for w in caught);warning_records=[{'category':k[0],'message':k[1],'count':v} for k,v in sorted(warning_counts.items())]
   def split_record(rel):
    if rel is None:return None
    if rel=='synthetic generated':return {'reference':rel,'seed_rule':'stable_seed(20260910, split family, dataset, chain, fraction, repeat)'}
    path=ROOT/rel;return {'reference':rel,'sha256':sha(path)}
   split_ref={'outer_registry':split_record('synthetic generated' if args.mode=='synthetic' else (f'data/04_splits/{dataset_id}_outer_assignments.csv.gz' if float(job['sample_fraction'])==1 else f'data/04_splits/{dataset_id}_sample_outer_assignments.csv.gz')),'C1_registry':split_record('synthetic generated' if args.mode=='synthetic' else f'data/04_splits/{dataset_id}_C1_assignments.csv.gz'),'sample_membership':split_record(None if float(job['sample_fraction'])==1 else f'data/04_splits/{dataset_id}_sample_chain_membership.csv.gz')}
   fit_rows=prov.pop('fit_source_rows',[]);prov['fit_source_row_count']=len(fit_rows);prov['fit_source_rows_sha256']=canonical_hash(fit_rows);prov['split_references']=split_ref
   pred=pred.copy();pred.insert(0,'run_id',job['run_id']);pred.insert(0,'fold',job['fold']);pred.insert(0,'repeat',job['repeat']);pred.insert(0,'chain',job['chain']);pred.insert(0,'sample_fraction',job['sample_fraction']);pred.insert(0,'phase',job['phase']);pred.insert(0,'condition',job['condition']);pred.insert(0,'model',job['model']);pred.insert(0,'dataset_id',job['dataset_id']);pred['selected_threshold']=met.get('threshold');pred['decision_score']=pred['probability']
   elapsed=time.perf_counter()-start;rss=peak_rss_bytes();registry_auth=job.get('authorization','NOT_APPLICABLE')
   runtime_scope=authorization.get('authorization_scope') if args.mode=='real' else 'SYNTHETIC_NOT_APPLICABLE'
   clean_job={k:v for k,v in job.items() if k!='authorization'}
   auth_meta={'registry_authorization_status':registry_auth,'runtime_authorization_scope':runtime_scope}
   summary=clean_job|auth_meta|{'pilot_only':args.mode=='synthetic','rows_in_analysis_frame':len(work),'train_rows':len(tr),'evaluation_rows':len(pred),'runtime_seconds':elapsed,'peak_rss_bytes':max(rss0,rss),'warning_count':sum(warning_counts.values()),'warning_categories':sorted({x[0] for x in warning_counts}),**met}
   manifest=clean_job|auth_meta|expected|{'pilot_only':args.mode=='synthetic','started_utc':attempt['utc'],'environment':{'python':sys.version,'platform':sys.platform},'data_reference_sha256':canonical_hash(work.__source_row__.astype(int).tolist()),'warnings':warning_records}
   store.save(job,summary,pred,prov,manifest);append_jsonl(TRACK/f'{args.mode}_attempt_log.jsonl',clean_job|auth_meta|{'event':'COMPLETE','utc':utc(),'runtime_seconds':elapsed});index.append(clean_job|{'status':'COMPLETE','output_directory':str(d)});mean_elapsed=(time.perf_counter()-sequence_start)/processed;eta=mean_elapsed*(total_jobs-processed);print(f'\r[PROGRESS] {processed}/{total_jobs} COMPLETE ({elapsed:.1f}s) | ETA {eta/60:.1f} min | {job["model"]} | {label}',end='',flush=True)
   if processed%25==0 or processed==total_jobs:print(flush=True)
  except Exception as e:
   print(f'\n[FAILED] {processed}/{total_jobs} | {job["model"]} | {label} | {e!r}',flush=True);append_jsonl(TRACK/f'{args.mode}_attempt_log.jsonl',job|{'event':'FAILED','utc':utc(),'error':repr(e),'traceback':traceback.format_exc()})
   infrastructure=isinstance(e,(OSError,TimeoutError,MemoryError))
   if infrastructure and not args.no_infrastructure_retry and int(job.get('_retry_count',0))<1:
    retry=dict(job);retry['_retry_count']=1;queue.append(retry);append_jsonl(TRACK/f'{args.mode}_attempt_log.jsonl',job|{'event':'RETRY_SCHEDULED','utc':utc(),'retry_number':1});continue
   index.append(job|{'status':'FAILED','error':repr(e),'output_directory':str(d)})
   if not args.continue_on_error:raise
 if current_model is not None:print(f'\n[MODEL COMPLETE] {dataset_id} / {current_model}',flush=True)
 out=TRACK/f'{dataset_id}_{args.mode}_{args.model}_{args.phase}_latest.json';atomic_json(out,{'dataset_id':dataset_id,'mode':args.mode,'created_utc':utc(),'jobs':index});print('\n[RUNNER COMPLETE]',flush=True);print(json.dumps({'dataset_id':dataset_id,'mode':args.mode,'models':models,'jobs':len(index),'status_counts':pd.Series([x['status'] for x in index]).value_counts().to_dict(),'index':str(out)},indent=2),flush=True);return index

if __name__=='__main__':raise SystemExit('Use a dataset-specific runner in execution/')
