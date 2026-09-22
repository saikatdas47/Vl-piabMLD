#!/usr/bin/env python3
"""Pre-experiment execution infrastructure. Scientific execution is gated off."""
from __future__ import annotations
import csv, hashlib, json, math, os, platform, sys, time, traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'data/00_tools/ml_vendor'
if sys.platform=='darwin' and VENDOR.exists():sys.path.insert(0,str(VENDOR))
import numpy as np, pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (average_precision_score,balanced_accuracy_score,brier_score_loss,f1_score,roc_auc_score)
from sklearn.model_selection import ParameterSampler,StratifiedGroupKFold,cross_val_predict
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from benchmark_pipeline import MODEL_SPACES,SPECIAL_COLUMNS,build_model,build_preprocessor,build_reference_pipeline,stable_seed

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path:Path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical_hash(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def select_threshold(y,p):
 vals=np.unique(np.r_[0.,p,1.]); candidates=(vals[:-1]+vals[1:])/2
 scored=[(balanced_accuracy_score(y,p>=t),-abs(t-.5),-t,t) for t in candidates]
 return float(max(scored)[3])
def calibration(y,p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6); x=np.log(p/(1-p))
 if np.std(x)==0:return math.nan,math.nan
 from sklearn.linear_model import LogisticRegression
 m=LogisticRegression(C=1e6,solver='lbfgs',max_iter=10000).fit(x.reshape(-1,1),y)
 return float(m.intercept_[0]),float(m.coef_[0,0])
def metrics(y,p,threshold):
 pred=np.asarray(p)>=threshold; ci,cs=calibration(y,p)
 return {'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),
 'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'F1':float(f1_score(y,pred)),
 'Brier':float(brier_score_loss(y,p)),'calibration_intercept':ci,'calibration_slope':cs,'threshold':float(threshold)}

def raw_feature_scores(X,y):
 vals=[]
 for c in X.columns:
  s=X[c]; num=pd.to_numeric(s,errors='coerce')
  if num.notna().mean()>=.98: z=num.fillna(num.median()).to_numpy(float)
  else:
   z=pd.Series(pd.factorize(s.astype('string').fillna('<MISSING>'),sort=True)[0],index=s.index).to_numpy(float)
  score=f_classif(z.reshape(-1,1),y)[0][0]; vals.append((c,-np.inf if not np.isfinite(score) else float(score)))
 return sorted(vals,key=lambda x:(-x[1],x[0]))
def choose_columns(X,y,percentile):
 ranked=raw_feature_scores(X,y); n=max(1,math.ceil(len(ranked)*percentile/100)); return [x[0] for x in ranked[:n]]

def inner_splits(y,g,seed,n=None):
 cfg=json.loads((ROOT/'config/experiment_config.json').read_text())
 n=int(n or cfg['inner_folds'])
 return list(StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,g))
def tune_pipeline(pipe,space,X,y,g,seed,n_iter,inner):
 best=None
 for params in ParameterSampler(space,n_iter=min(n_iter,max(1,int(np.prod([len(v) for v in space.values()])))),random_state=seed):
  scores=[]
  for tr,va in inner:
   q=clone(pipe).set_params(**params);q.fit(X.iloc[tr] if hasattr(X,'iloc') else X[tr],y[tr]);p=q.predict_proba(X.iloc[va] if hasattr(X,'iloc') else X[va])[:,1];scores.append(roc_auc_score(y[va],p))
  row=(float(np.mean(scores)),json.dumps(params,sort_keys=True,default=str),params)
  if best is None or row[:2]>best[:2]:best=row
 return best[2],best[0]
def condition_space(model_name,condition):
 space=dict(MODEL_SPACES[model_name])
 if condition=='C4':space.pop('feature_selection__percentile',None)
 return space
def oof_threshold(pipe,X,y,inner):
 p=cross_val_predict(pipe,X,y,cv=inner,method='predict_proba',n_jobs=1)[:,1];return select_threshold(y,p)

def global_preprocess(frame,X,model_name):
 """Fit the exact reference preprocessor on the complete cohort for C3."""
 preprocessor=build_preprocessor(frame,model_name)
 return preprocessor.fit_transform(X),preprocessor

@dataclass
class RunStore:
 root:Path
 def paths(self,run_id):
  d=self.root/run_id;return d,d/'summary.json',d/'predictions.csv',d/'manifest.json'
 def complete(self,run_id,expected=None):
  d,s,p,m=self.paths(run_id)
  if not all(x.exists() for x in [s,p,m]):return False
  z=json.loads(m.read_text())
  if expected and any(z.get(k)!=v for k,v in expected.items()):return False
  return z.get('status')=='COMPLETE' and sha(s)==z['summary_sha256'] and sha(p)==z['predictions_sha256']
 def save(self,run_id,summary,preds,manifest):
  d,s,p,m=self.paths(run_id);d.mkdir(parents=True,exist_ok=True)
  if any(x.exists() for x in [s,p,m]):raise RuntimeError('partial or mismatched run exists; refusing overwrite')
  st=s.with_suffix('.tmp');pt=p.with_suffix('.tmp');mt=m.with_suffix('.tmp')
  st.write_text(json.dumps(summary,indent=2,default=str)+'\n');preds.to_csv(pt,index=False)
  manifest|={'status':'COMPLETE','summary_sha256':sha(st),'predictions_sha256':sha(pt),'completed_utc':utc()};mt.write_text(json.dumps(manifest,indent=2)+'\n')
  os.replace(st,s);os.replace(pt,p);os.replace(mt,m)

def execute_outer(frame,train,test,condition,model_name,dataset_id,repeat,fold,n_iter=2,source_folds=None):
 X=frame.drop(columns=list(SPECIAL_COLUMNS),errors='ignore');y=frame.__target__.astype(int).to_numpy();g=frame.__group_id__.astype(str).to_numpy()
 model_seed=stable_seed(20260910,'model',dataset_id,model_name,condition,repeat,fold); search_seed=stable_seed(20260910,'search',dataset_id,model_name,condition,repeat,fold); inn_seed=stable_seed(20260910,'inner',dataset_id,repeat,fold)
 fit_ids=set(frame.__source_row__.iloc[train].astype(int)); contamination=[]; copy_provenance=[]; selected=list(X.columns)
 if condition=='C2':
  all_idx=np.arange(len(frame));pipe=build_reference_pipeline(frame,model_name,model_seed,100);inner=inner_splits(y,g,stable_seed(20260910,'nonnested',dataset_id,repeat))
  best,score=tune_pipeline(pipe,condition_space(model_name,condition),X,y,g,search_seed,n_iter,inner);pipe.set_params(**best)
  prob=cross_val_predict(pipe,X,y,cv=inner,method='predict_proba',n_jobs=1)[:,1];threshold=select_threshold(y,prob);met=metrics(y,prob,threshold)
  pred=pd.DataFrame({'source_row':frame.__source_row__.to_numpy(),'group_id':g,'true_label':y,'probability':prob,'binary_prediction':(prob>=threshold).astype(int)})
  return met|{'best_params':best,'inner_cv_AUROC':score},pred,{'fit_source_rows':sorted(frame.__source_row__.astype(int)),'contaminating_test_source_indices':[],'copy_provenance':[],'model_seed':model_seed,'search_seed':search_seed,'inner_seed':inn_seed,'selected_raw_columns':selected,'reported_from_same_folds_used_for_selection':True}
 if condition=='C3':
  from sklearn.feature_selection import SelectPercentile
  from sklearn.feature_selection import VarianceThreshold
  Z,global_object=global_preprocess(frame,X,model_name); Xt,Xe=Z[train],Z[test]; pipe=Pipeline([('constant_filter',VarianceThreshold(0.0)),('feature_selection',SelectPercentile(score_func=f_classif,percentile=100)),('oversample',RandomOverSampler(random_state=model_seed)),('model',build_model(model_name,model_seed))])
 elif condition=='C4':
  selected=choose_columns(X,y,50);Xt,Xe=X.iloc[train][selected],X.iloc[test][selected]
  # Rebuild against the selected raw columns only.
  local=frame.iloc[train][selected+['__source_row__','__group_id__','__target__']];pipe=build_reference_pipeline(local,model_name,model_seed,100);pipe.set_params(feature_selection__percentile=100)
 elif condition=='C5':
  ros=RandomOverSampler(random_state=stable_seed(20260910,'oversample',dataset_id,repeat));idx=np.arange(len(frame)).reshape(-1,1);ri,_=ros.fit_resample(idx,y);ri=ri.ravel();copies=ri[len(frame):]; contamination=[int(i) for i in copies if i in set(test)]
  fold_lookup=np.asarray(source_folds) if source_folds is not None else np.full(len(frame),-1)
  copy_provenance=[{'copy_id':f'{dataset_id}_r{repeat}_copy{k}','source_row':int(frame.__source_row__.iloc[i]),'source_group_id':str(g[i]),'source_outer_fold':int(fold_lookup[i]),'destination_outer_fold':int(fold),'repeat':int(repeat),'sampler_seed':stable_seed(20260910,'oversample',dataset_id,repeat)} for k,i in enumerate(copies)]
  aug=np.r_[train,copies];Xt,Xe=X.iloc[aug].reset_index(drop=True),X.iloc[test];yt=y[aug];gt=np.array([g[i] for i in aug]);fit_ids|=set(frame.__source_row__.iloc[copies].astype(int));pipe=build_reference_pipeline(frame.iloc[aug],model_name,model_seed,100)
  pipe=Pipeline([(name,step) for name,step in pipe.steps if name!='oversample']);train, y_train, g_train=aug,yt,gt
 elif condition=='C6':
  Xt,Xe=X.iloc[train],X.iloc[test];pipe=build_reference_pipeline(frame.iloc[train],model_name,model_seed,100);inner=None;best={};score=math.nan
 else:
  Xt,Xe=X.iloc[train],X.iloc[test];pipe=build_reference_pipeline(frame.iloc[train],model_name,model_seed,100)
 ytr=y[train] if condition!='C5' else y_train;gtr=g[train] if condition!='C5' else g_train
 if condition!='C6':
  inner=inner_splits(ytr,gtr,inn_seed);best,score=tune_pipeline(pipe,condition_space(model_name,condition),Xt,ytr,gtr,search_seed,n_iter,inner);pipe.set_params(**best)
  threshold=oof_threshold(pipe,Xt,ytr,inner)
 else: threshold=.5
 pipe.fit(Xt,ytr);prob=pipe.predict_proba(Xe)[:,1];met=metrics(y[test],prob,threshold)
 pred=pd.DataFrame({'source_row':frame.__source_row__.iloc[test].to_numpy(),'group_id':g[test],'true_label':y[test],'probability':prob,'binary_prediction':(prob>=threshold).astype(int)})
 global_operation={'C3':'preprocessor','C4':'supervised_feature_selection','C5':'random_oversampler'}.get(condition)
 global_rows=sorted(frame.__source_row__.astype(int)) if global_operation else []
 return met|{'best_params':best,'inner_cv_AUROC':score},pred,{'fit_source_rows':sorted(fit_ids),'global_operation':global_operation,'global_fit_source_row_count':len(global_rows),'global_fit_source_rows_sha256':canonical_hash(global_rows) if global_rows else None,'contaminating_test_source_indices':contamination,'copy_provenance':copy_provenance,'model_seed':model_seed,'search_seed':search_seed,'inner_seed':inn_seed,'selected_raw_columns':selected}

def hierarchical_subsets(frame,fractions,chains,dataset_id,min_rows=200,min_minority=50):
 rows=[]; groups=frame.groupby('__group_id__')['__target__'].agg(['count','sum']).reset_index()
 for chain in range(chains):
  order=groups.sample(frac=1,random_state=stable_seed(20260910,'subsample',dataset_id,chain)).__group_id__.tolist()
  for frac in sorted(fractions,reverse=True):
   keep=set(order[:max(1,math.ceil(len(order)*frac))]);sub=frame[frame.__group_id__.isin(keep)];counts=sub.__target__.value_counts();ok=len(sub)>=min_rows and len(counts)==2 and counts.min()>=min_minority
   rows.append({'dataset_id':dataset_id,'chain':chain,'fraction':frac,'rows':len(sub),'groups':len(keep),'negative':int(counts.get(0,0)),'positive':int(counts.get(1,0)),'status':'ESTIMABLE' if ok else 'NOT_ESTIMABLE','seed':stable_seed(20260910,'subsample',dataset_id,chain)})
 return pd.DataFrame(rows)

def environment_manifest():
 import numpy,sklearn,scipy
 return {'created_utc':utc(),'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__}
