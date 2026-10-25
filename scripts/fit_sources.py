"""Optional source-only fits. Does not run unless explicitly called."""
from pathlib import Path
import argparse,json,sys,os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'2')
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adsorption_transfer.features import FEATURES
from adsorption_transfer.source_models import SOURCE_IDS,make_model,fit_with_groups,q_predictions

def fit_one(data,name,target):
    source=data[data.material_group!=target].reset_index(drop=True)
    pool=data[data.material_group==target].copy();oof=np.empty(len(source));folds=np.full(len(source),-1);epochs=[]
    for fold,(train,held) in enumerate(GroupKFold(3).split(source,groups=source.material_group)):
        tr=source.iloc[train];va=source.iloc[held]
        assert set(tr.material_group).isdisjoint(va.material_group) and target not in set(tr.material_group)
        if name=='FT':
            from adsorption_transfer.transformer import DEFAULT_CONFIG,fit_model,predict_log
            it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=3407+fold).split(tr,groups=tr.material_group))
            _,_,_,epoch,_=fit_model(tr.iloc[it],DEFAULT_CONFIG,3407+fold,validation=tr.iloc[iv])
            model,enc,_,_,_=fit_model(tr,DEFAULT_CONFIG,3507+fold,fixed_epochs=epoch)
            oof[held]=np.exp(np.clip(predict_log(model,enc,va),-30,30));epochs.append(epoch)
        else:
            model=make_model(name);fit_with_groups(model,tr[FEATURES],np.log(tr.qe),tr.material_group)
            oof[held]=q_predictions(model,va[FEATURES])
        folds[held]=fold
    if name=='FT':
        model,enc,_,_,_=fit_model(source,DEFAULT_CONFIG,4407,fixed_epochs=max(1,int(np.median(epochs))))
        pred=np.exp(np.clip(predict_log(model,enc,pool),-30,30))
    else:
        model=make_model(name);fit_with_groups(model,source[FEATURES],np.log(source.qe),source.material_group)
        pred=q_predictions(model,pool[FEATURES])
    a=pool[['excel_row','material_group','qe']].copy();a['target_material']=target;a['q0']=pred
    b=source[['excel_row','material_group','qe']].copy();b['target_material']=target;b['q0']=oof;b['inner_fold']=folds
    return a,b

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--models',nargs='+',choices=SOURCE_IDS,default=['Ridge']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=pd.read_csv(a.input)
    for name in a.models:
        if name=='FT':
            import torch
            torch.set_num_threads(2)
            try:torch.set_num_interop_threads(1)
            except RuntimeError:pass
        folder=a.output/name
        if folder.exists():raise FileExistsError('Choose a new output folder; existing source fits are never overwritten: '+str(folder))
        folder.mkdir(parents=True);pools=[];oofs=[]
        for target in sorted(data.material_group.unique()):
            aa,bb=fit_one(data,name,target);pools.append(aa);oofs.append(bb);print(name,target,flush=True)
        pd.concat(pools,ignore_index=True).to_csv(folder/'pool_predictions.csv',index=False)
        pd.concat(oofs,ignore_index=True).to_csv(folder/'source_oof_predictions.csv',index=False)
        (folder/'configuration.json').write_text(json.dumps({'source':name,'source_seed':3407,'target_labels_used':False,'features':FEATURES},indent=2))
