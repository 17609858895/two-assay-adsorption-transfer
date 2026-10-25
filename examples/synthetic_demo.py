"""A small synthetic algebra/information-timing demonstration, not paper data."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.special import expit,logit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adsorption_transfer.features import FEATURES
from adsorption_transfer.methods import EmpiricalPrior,BudgetedLabels
from adsorption_transfer.protocol import prepare_protocol,check_protocol
from adsorption_transfer.design import pair_risks,check_pair,choose_batch_pair
from adsorption_transfer.mixtures import SourceMixturePrior
from adsorption_transfer.calibration import METHODS,predict
from adsorption_transfer.influence import operators

def synthetic_data(seed=711):
    rng=np.random.default_rng(seed);rows=[]
    for material in range(6):
        for block in range(7):
            for repeat in range(3):
                row={name:float(rng.uniform(.5,4)) for name in FEATURES}
                row.update(Feedstock='Feedstock '+str(material%3),inorganics='Ion '+str(block%2),Anion_type='Anion '+str(block%2))
                row['Ci']=float(np.exp(rng.uniform(0,4)));row['dose_calculated']=float(np.exp(rng.uniform(-1,2)))
                row['available_mass_per_g']=row['Ci']/row['dose_calculated']
                row['Adsorption_time (min)']=[30.,60.,120.][repeat];row['solution pH']=4.+block*.4
                row['Volume (L)']=.05;row['loading (g)']=.05*row['dose_calculated']
                r0=expit(-.6+.13*np.log(row['Ci'])-.18*np.log(row['dose_calculated']))
                delta=(material-2)*.11+.10*np.log(row['available_mass_per_g'])+.06*np.sin(block)
                row.update(excel_row=len(rows)+2,material_group='M'+str(material),condition_block=material*7+block,duplicate_group=len(rows),q0=row['available_mass_per_g']*r0,qe=row['available_mass_per_g']*expit(logit(r0)+delta))
                rows.append(row)
    return pd.DataFrame(rows)

def example_context():
    data=synthetic_data();roles,summary,_=prepare_protocol(data,seed=3407,repeats=1)
    assert check_protocol(roles,summary).all_checks_pass.all()
    target='M5';source=data[data.material_group!=target]
    prior=EmpiricalPrior(source,source.q0.to_numpy(),'logit')
    z=roles[roles.target_material==target]
    indexed=data.set_index('excel_row',drop=False)
    ci=z[z.role=='candidate'].excel_row.to_numpy();qi=z[z.role=='query'].excel_row.to_numpy()
    candidates=indexed.loc[ci].drop(columns=['qe','q0']);query=indexed.loc[qi].drop(columns=['qe','q0'])
    qc=indexed.loc[ci,'q0'].to_numpy();qq=indexed.loc[qi,'q0'].to_numpy()
    return data,source,prior,candidates,qc,query,qq,indexed.loc[ci,'qe'].to_numpy()

def run():
    data,source,prior,candidates,qc,query,qq,outcomes=example_context()
    ii,jj,risk,objectives,terms=pair_risks(candidates,qc,query,qq,prior)
    checks=check_pair(candidates,qc,query,qq,prior,ii,jj,risk,terms)
    chosen,diagnostic=choose_batch_pair('simple_convex_minimax',candidates,qc,query,qq,prior)
    # Selection above has no object through which to access the candidate outcomes.
    oracle=BudgetedLabels(candidates.excel_row,candidates.condition_block,outcomes,max_k=2)
    revealed=np.asarray([oracle.reveal(j) for j in chosen]);anchors=candidates.iloc[chosen];qa=qc[chosen]
    priors={'logit':prior,'ln':EmpiricalPrior(source,source.q0.to_numpy(),'ln')}
    mixture=SourceMixturePrior(source,source.q0.to_numpy(),prior)
    predictions={method:predict(method,query,qq,anchors,qa,revealed,priors,mixture) for method in METHODS}
    for value in predictions.values():assert np.isfinite(value).all() and (value>0).all() and (value<=query.available_mass_per_g).all()
    W,_=operators(query,anchors,qq,qa,prior)['Convex_contrast']
    report={'data':'synthetic','source_materials':source.material_group.nunique(),'queries':len(query),'legal_pairs':len(ii),'selected_excel_rows':anchors.excel_row.astype(int).tolist(),'revealed_outcomes':len(oracle.access),'calibrators':len(predictions),'minimum_convex_weight':float(W.min()),'maximum_latent_l1_influence':float(np.abs(W).sum(1).max()),'maximum_direct_risk_check_error':max(c['matrix_risk_error'] for c in checks),'all_predictions_inside_supply':True}
    return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
