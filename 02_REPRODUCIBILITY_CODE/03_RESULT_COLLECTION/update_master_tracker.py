#!/usr/bin/env python3
"""Regenerate tracker tables from immutable run manifests; never edits results."""
from pathlib import Path
import hashlib,json,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/07_execution_tracking';OUT.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'config/experiment_config.json').read_text());primary=pd.read_csv(ROOT/'data/06_preexecution/primary_job_registry.csv');sample=pd.read_csv(ROOT/'data/06_preexecution/sample_size_job_registry.csv');planned=pd.concat([primary,sample],ignore_index=True)
current_config_sha=hashlib.sha256((ROOT/'config/experiment_config.json').read_bytes()).hexdigest()
def identity(row):
 return (str(row['dataset_id']),str(row['model']),float(row['sample_fraction']),str(row['chain']),str(row['condition']),int(row['repeat']),str(row['fold']))
rows=[]
for (dataset,model),g in planned.groupby(['dataset_id','model'],sort=True):
 base=ROOT/'results'/dataset/model;manifests=list(base.rglob('manifest.json')) if base.exists() else [];complete_ids=set();duplicate_current_ids=set();failed=0;latest='';planned_ids={identity(r) for _,r in g.iterrows()}
 for p in manifests:
  try:
   m=json.loads(p.read_text())
   if m.get('config_sha256')!=current_config_sha:continue
   mid=identity(m)
   if mid not in planned_ids:failed+=1;continue
   if m.get('status')=='COMPLETE':
    if mid in complete_ids:duplicate_current_ids.add(mid)
    complete_ids.add(mid);latest=max(latest,m.get('completed_utc',''))
   else:failed+=1
  except Exception:failed+=1
 failed+=len(duplicate_current_ids);complete=len(complete_ids)
 status='FAILED_ATTENTION' if failed else 'COMPLETE' if complete==len(planned_ids) else 'IN_PROGRESS' if complete else 'NOT_STARTED'
 rows.append({'dataset_id':dataset,'model':model,'planned_atomic_units':len(planned_ids),'completed_atomic_units':complete,'failed_atomic_units':failed,'remaining_atomic_units':len(planned_ids)-complete,'percent_complete':complete/len(planned_ids),'status':status,'last_completion_utc':latest,'result_root':str(base.relative_to(ROOT))})
tracker=pd.DataFrame(rows);tracker.to_csv(OUT/'master_execution_tracker.csv',index=False)
u=shutil.disk_usage(ROOT);result_bytes=sum(p.stat().st_size for p in (ROOT/'results').rglob('*') if p.is_file()) if (ROOT/'results').exists() else 0;free=u.free/(1024**3);pol=cfg['storage_policy'];drive_fuse=str(ROOT).startswith('/content/drive/');reliable=not drive_fuse
state='CAUTION' if drive_fuse else 'STOP' if free<pol['minimum_free_gib_before_new_atomic_run'] else 'CAUTION' if free<pol['warning_free_gib'] else 'PASS'
storage={'created_utc':pd.Timestamp.utcnow().isoformat(),'status':state,'free_gib':round(free,3),'used_gib':round(u.used/(1024**3),3),'total_gib':round(u.total/(1024**3),3),'capacity_reading_reliable':reliable,'capacity_note':'Google Drive FUSE capacity is informational and is not the account quota; actual write failures remain explicit and resumable.' if drive_fuse else None,'results_bytes':result_bytes,'minimum_free_gib':pol['minimum_free_gib_before_new_atomic_run'],'warning_free_gib':pol['warning_free_gib'],'automatic_deletion':False}
(OUT/'storage_status.json').write_text(json.dumps(storage,indent=2)+'\n');print(json.dumps({'tracker_rows':len(tracker),'status_counts':tracker.status.value_counts().to_dict(),'storage':storage},indent=2))
