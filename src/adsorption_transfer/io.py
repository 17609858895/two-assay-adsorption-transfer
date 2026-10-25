"""Validate source-only OOF and target-pool prediction files."""
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import positive

def get_predictions(folder,d):
    folder=Path(folder);pool=pd.read_csv(folder/'pool_predictions.csv');oof=pd.read_csv(folder/'source_oof_predictions.csv')
    for z in [pool,oof]:
        if 'q0' not in z:
            cols=[c for c in z if c.startswith('q0_')]
            if len(cols)!=1:raise ValueError('need one q0 column: '+str(folder))
            z['q0']=z[cols[0]]
        z['q0']=positive(z.q0)
    materials=set(d.material_group.unique())
    assert set(pool.target_material)==materials and set(oof.target_material)==materials
    assert len(pool)==len(d) and not pool.duplicated(['target_material','excel_row']).any()
    checks=[]
    for material in sorted(materials):
        p=pool[pool.target_material==material];z=oof[oof.target_material==material]
        target=set(d.loc[d.material_group==material,'excel_row']);source=set(d.loc[d.material_group!=material,'excel_row'])
        assert set(p.excel_row)==target and set(z.excel_row)==source and z.excel_row.is_unique
        assert np.allclose(d.loc[p.excel_row,'qe'],p.qe) and np.allclose(d.loc[z.excel_row,'qe'],z.qe)
        assert not (d.loc[z.excel_row,'material_group']==material).any()
        if 'inner_fold' not in z:raise AssertionError('OOF fold provenance absent')
        zz=z.copy();zz['material_group']=d.loc[z.excel_row,'material_group'].to_numpy()
        assert zz.groupby('material_group').inner_fold.nunique().max()==1
        assert zz.inner_fold.nunique()==3
        checks.append(dict(target_material=material,n_target=len(p),n_source_oof=len(z),n_source_materials=zz.material_group.nunique(),
                           source_excludes_target=True,source_material_grouped_oof=True,n_inner_folds=3))
    return pool,oof,checks
