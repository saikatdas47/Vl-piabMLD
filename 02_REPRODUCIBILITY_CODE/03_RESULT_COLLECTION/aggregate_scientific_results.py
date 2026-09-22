#!/usr/bin/env python3
"""Validate immutable results and create prespecified analysis tables.

Default execution refuses incomplete archives. It never fits a model.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'data/00_tools/ml_vendor'
if sys.platform=='darwin' and VENDOR.exists():sys.path.insert(0,str(VENDOR))
import numpy as np,pandas as pd
from scipy.stats import friedmanchisquare,wilcoxon

METRICS=['AUROC','AUPRC','balanced_accuracy','F1','Brier','calibration_intercept','calibration_slope']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def holm(pvals):
 items=sorted(pvals.items(),key=lambda x:x[1]);m=len(items);out={};running=0
 for rank,(name,p) in enumerate(items):running=max(running,min(1,(m-rank)*p));out[name]=running
 return out
def bootstrap_median(values,n,seed):
 x=np.asarray(values,float);rng=np.random.default_rng(seed);boots=np.median(rng.choice(x,(n,len(x)),replace=True),axis=1);return float(np.median(x)),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))
def signed_rank(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return math.nan,math.nan
 if np.allclose(x,0):return 0.0,1.0
 z=wilcoxon(x,alternative='two-sided',zero_method='wilcox');return float(z.statistic),float(z.pvalue)
def collect(root):
 rows=[];bad=[]
 for mpath in Path(root).rglob('manifest.json'):
  try:
   m=json.loads(mpath.read_text());d=mpath.parent;s=d/'summary.json';p=d/'predictions.csv.gz';v=d/'provenance.json.gz'
   if m.get('status')!='COMPLETE' or sha(s)!=m['summary_sha256'] or sha(p)!=m['predictions_sha256'] or sha(v)!=m['provenance_sha256']:bad.append(str(d));continue
   rows.append(json.loads(s.read_text()))
  except Exception as e:bad.append(f'{mpath}: {e!r}')
 return pd.DataFrame(rows),bad
def analyze(raw,cfg):
 keys=['dataset_id','model','condition','sample_fraction','chain'];agg=raw.groupby(keys,dropna=False)[METRICS].agg(['mean','median']).reset_index();agg.columns=['_'.join(x).rstrip('_') for x in agg.columns]
 full=agg[agg.sample_fraction==1].copy();base=full[full.condition=='C0'].set_index(['dataset_id','model'])
 pairs=[]
 for _,r in full[full.condition!='C0'].iterrows():
  key=(r.dataset_id,r.model)
  if key not in base.index:continue
  q={'dataset_id':r.dataset_id,'model':r.model,'condition':r.condition,'sample_fraction':1.0,'chain':r.chain}
  for metric in METRICS:
   q[f'delta_{metric}']=r[f'{metric}_mean']-base.loc[key,f'{metric}_mean'];q[f'delta_median_{metric}']=r[f'{metric}_median']-base.loc[key,f'{metric}_median']
  pairs.append(q)
 paired=pd.DataFrame(pairs);dataset=paired.groupby(['dataset_id','condition'],dropna=False).agg({c:'mean' for c in paired.columns if c.startswith('delta_')}).reset_index() if len(paired) else pd.DataFrame()
 tests=[];pvals={};nboot=cfg['statistical_analysis']['bootstrap_resamples'];bseed=cfg['statistical_analysis']['bootstrap_seed']
 for i,c in enumerate(['C2','C3','C4','C5']):
  x=dataset[dataset.condition==c].delta_AUROC.dropna().to_numpy() if len(dataset) else np.array([])
  if len(x)>=cfg['statistical_analysis']['minimum_complete_datasets_for_confirmatory_test']:
   stat,p=signed_rank(x);med,lo,hi=bootstrap_median(x,nboot,bseed+i);pvals[c]=float(p);tests.append({'contrast':f'{c}-C0','datasets':len(x),'median_delta_AUROC':med,'ci95_low':lo,'ci95_high':hi,'wilcoxon_statistic':stat,'p_value':p})
  else:tests.append({'contrast':f'{c}-C0','datasets':len(x),'status':'INSUFFICIENT_COMPLETE_DATASETS'})
 adj=holm(pvals)
 for r in tests:
  c=r['contrast'].split('-')[0]
  if c in adj:r['holm_adjusted_p']=adj[c]
 friedman={"status":"NOT_ESTIMABLE"}
 if len(dataset):
  wide=dataset[dataset.condition.isin(['C2','C3','C4','C5'])].pivot(index='dataset_id',columns='condition',values='delta_AUROC').dropna()
  if len(wide)>=cfg['statistical_analysis']['minimum_complete_datasets_for_confirmatory_test']:
   z=friedmanchisquare(np.zeros(len(wide)),*[wide[c].to_numpy() for c in ['C2','C3','C4','C5']]);friedman={'status':'ESTIMATED','datasets':len(wide),'statistic':float(z.statistic),'p_value':float(z.pvalue)}
 sensitivity=[]
 for i,c in enumerate(['C2','C3','C4','C5']):
  x=dataset[dataset.condition==c].delta_median_AUROC.dropna().to_numpy() if len(dataset) else np.array([])
  if len(x)>=cfg['statistical_analysis']['minimum_complete_datasets_for_confirmatory_test']:
   stat,p=signed_rank(x);med,lo,hi=bootstrap_median(x,nboot,bseed+100+i);sensitivity.append({'contrast':f'{c}-C0','datasets':len(x),'aggregation':'median','median_delta_AUROC':med,'ci95_low':lo,'ci95_high':hi,'wilcoxon_statistic':stat,'p_value':p})
  else:sensitivity.append({'contrast':f'{c}-C0','datasets':len(x),'aggregation':'median','status':'INSUFFICIENT_COMPLETE_DATASETS'})
 sample=agg[agg.sample_fraction<1].copy();sample_pairs=[]
 if len(sample):
  sb=sample[sample.condition=='C0'].set_index(['dataset_id','model','sample_fraction','chain'])
  for _,r in sample[sample.condition!='C0'].iterrows():
   key=(r.dataset_id,r.model,r.sample_fraction,r.chain)
   if key not in sb.index:continue
   q={'dataset_id':r.dataset_id,'model':r.model,'sample_fraction':r.sample_fraction,'chain':r.chain,'condition':r.condition}
   for metric in METRICS:q[f'delta_{metric}']=r[f'{metric}_mean']-sb.loc[key,f'{metric}_mean']
   sample_pairs.append(q)
 return agg,paired,dataset,pd.DataFrame(tests),pd.DataFrame(sensitivity),pd.DataFrame(sample_pairs),friedman
def self_test():
 rows=[]
 for d in range(15):
  for m in ['a','b','c','d','e']:
   for c,shift in [('C0',0),('C2',.01),('C3',.02),('C4',.03),('C5',.04),('C6',0)]:
    for fold in range(3):rows.append({'dataset_id':f'D{d:03d}','model':m,'condition':c,'sample_fraction':1.0,'chain':'PRIMARY','AUROC':.65+shift+d/10000,'AUPRC':.5+shift,'balanced_accuracy':.6+shift,'F1':.55+shift,'Brier':.2-shift,'calibration_intercept':0,'calibration_slope':1})
 cfg=json.loads((ROOT/'config/experiment_config.json').read_text());a,p,d,t,s,z,f=analyze(pd.DataFrame(rows),cfg);assert len(t)==4 and len(s)==4 and all(t.datasets==15) and f['status']=='ESTIMATED';print(json.dumps({'status':'PASS','fold_aggregates':len(a),'paired_cells':len(p),'dataset_condition_cells':len(d),'primary_tests':len(t),'median_sensitivity_tests':len(s)}));return
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results-root',default=str(ROOT/'results'));ap.add_argument('--output',default=str(ROOT/'results/_combined_analysis'));ap.add_argument('--allow-incomplete',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 if args.self_test:return self_test()
 cfg=json.loads((ROOT/'config/experiment_config.json').read_text());raw,bad=collect(args.results_root);tracker=pd.read_csv(ROOT/'data/07_execution_tracking/master_execution_tracker.csv')
 if (bad or not tracker.status.eq('COMPLETE').all()) and not args.allow_incomplete:raise RuntimeError(f'Archive incomplete: invalid={len(bad)}, complete packages={(tracker.status=="COMPLETE").sum()}/75')
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);agg,pair,dataset,tests,sensitivity,sample_pairs,friedman=analyze(raw,cfg)
 for name,df in [('validated_atomic_summaries',raw),('fold_aggregates',agg),('paired_model_effects',pair),('dataset_condition_effects',dataset),('confirmatory_tests',tests),('median_aggregation_sensitivity_tests',sensitivity),('sample_size_condition_effects',sample_pairs)]:df.to_csv(out/f'{name}.csv',index=False)
 (out/'analysis_manifest.json').write_text(json.dumps({'scientific_archive_complete':bool(tracker.status.eq('COMPLETE').all()),'invalid_manifests':bad,'friedman':friedman,'configuration':cfg['statistical_analysis']},indent=2)+'\n')
if __name__=='__main__':main()
