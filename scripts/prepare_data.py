"""Prepare the modelling table from the separately obtained source workbook."""
from pathlib import Path
import argparse,hashlib
import numpy as np
import pandas as pd

def prepare(path):
    data=pd.read_excel(path,sheet_name='HMIs').dropna(axis=1,how='all')
    data.insert(0,'excel_row',np.arange(len(data))+2)
    original=[c for c in data if c!='excel_row']
    data['duplicate_group']=pd.factorize(pd.util.hash_pandas_object(data[original],index=False))[0]
    data['material_group']=data['Adsorbent'].astype(str)
    data['dose_calculated']=data['loading (g)']/data['Volume (L)']
    data['available_mass_per_g']=data['Ci']/data['dose_calculated']
    data['removal_fraction']=data['qe']/data['available_mass_per_g']
    block=['Adsorbent','inorganics','solution pH','adsorption_temp','rpm','Ion Concentration (M)','Anion_type','DOM']
    data['condition_block']=pd.factorize(pd.util.hash_pandas_object(data[block],index=False))[0]
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-other-workbook',action='store_true');a=p.parse_args()
    observed=hashlib.sha256(a.input.read_bytes()).hexdigest()
    expected='2074a592c184aa680f321d5d28f4c8371dba1c249c915a67352cc7e2e4fc725d'
    if observed!=expected and not a.allow_other_workbook:raise ValueError('Workbook differs from the study source; inspect it or explicitly use --allow-other-workbook')
    d=prepare(a.input);a.output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.output,index=False)
    print('Prepared rows',len(d),'material labels',d.material_group.nunique(),'condition blocks',d.condition_block.nunique())
