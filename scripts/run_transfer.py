"""Evaluate matched source/calibration/design configurations on fixed query rows."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adsorption_transfer.methods import EmpiricalPrior,BudgetedLabels,select_anchors
from adsorption_transfer.protocol import prepare_protocol,check_protocol
from adsorption_transfer.design import BATCH_POLICIES,choose_batch_pair
from adsorption_transfer.mixtures import SourceMixturePrior
from adsorption_transfer.calibration import METHODS,predict
from adsorption_transfer.metrics import ADDITIVE,metric_sums,derive
from adsorption_transfer.io import get_predictions

def evaluate(data,source_folder):
    d=data.set_index('excel_row',drop=False);roles,splits,_=prepare_protocol(data,seed=3407,repeats=5)
    assert check_protocol(roles,splits).all_checks_pass.all()
    pool,oof,_=get_predictions(source_folder,d);stats=[];anchors=[]
    for material in sorted(d.material_group.unique()):
        pp=pool[pool.target_material==material].set_index('excel_row')
        oo=oof[oof.target_material==material].set_index('excel_row');source=d.loc[oo.index]
        priors={sp:EmpiricalPrior(source,oo.q0.to_numpy(),sp) for sp in ['ln','logit']}
        mix=SourceMixturePrior(source,oo.q0.to_numpy(),priors['logit'])
        for scene,z in roles[roles.target_material==material].groupby('scene_id'):
            ci=z[z.role=='candidate'].excel_row.to_numpy();qi=z[z.role=='query'].excel_row.to_numpy()
            cx=d.loc[ci].drop(columns=['qe','Cf','removal_fraction'],errors='ignore');x=d.loc[qi].drop(columns=['qe','Cf','removal_fraction'],errors='ignore')
            qc=pp.loc[ci,'q0'].to_numpy();qq=pp.loc[qi,'q0'].to_numpy();s=x.available_mass_per_g.to_numpy()
            for policy in ['maximin','IVR_link_kernel']+BATCH_POLICIES:
                oracle=BudgetedLabels(ci,cx.condition_block.to_numpy(),d.loc[ci,'qe'].to_numpy(),5 if policy in ['maximin','IVR_link_kernel'] else 2)
                if policy in BATCH_POLICIES:
                    chosen,_=choose_batch_pair(policy,cx,qc,x,qq,priors['logit'])
                    yy=np.asarray([oracle.reveal(j) for j in chosen]);budgets=[2]
                else:
                    chosen,yy,_=select_anchors(policy,cx,qc,x,qq,priors,oracle,seed=int(z['seed'].iloc[0]),max_k=5);budgets=[2,5]
                for budget in budgets:
                    ids=chosen[:budget];aa=cx.iloc[ids];qa=qc[ids];ya=yy[:budget]
                    for step,index in enumerate(ids,1):anchors.append({'policy':policy,'budget':budget,'target_material':material,'scene_id':scene,'step':step,'excel_row':int(ci[index])})
                    for method in METHODS:
                        value=predict(method,x,qq,aa,qa,ya,priors,mix)
                        assert np.isfinite(value).all() and (value>0).all() and (value<=s*(1+1e-8)).all()
                        stats.append({'policy':policy,'budget':budget,'method':method,'target_material':material,'scene_id':scene,**derive(metric_sums(d.loc[qi,'qe'],value,s))})
    return pd.DataFrame(stats),pd.DataFrame(anchors)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--source-folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output folder: '+str(a.output))
    data=pd.read_csv(a.input);scene,anchors=evaluate(data,a.source_folder)
    summary=pd.DataFrame([{**dict(zip(['policy','budget','method'],key)),**derive(z[ADDITIVE].sum().to_dict())} for key,z in scene.groupby(['policy','budget','method'],sort=False)])
    a.output.mkdir(parents=True);scene.to_csv(a.output/'scene_metrics.csv',index=False);summary.to_csv(a.output/'summary.csv',index=False);anchors.to_csv(a.output/'anchors.csv',index=False)
    print('Configurations',len(summary),'scene rows',len(scene))
