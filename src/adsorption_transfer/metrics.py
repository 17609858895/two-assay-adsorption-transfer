"""Additive statistics and paired fixed-prediction material resampling."""
import numpy as np
import pandas as pd
ADDITIVE=['n','sse_q','sse_ln','sae_q','sae_ln','sum_y','sum_y2','sum_lny','sum_lny2','n_violation']

def positive(q):
    q=np.asarray(q,dtype=float)
    if not np.isfinite(q).all():raise AssertionError('nonfinite source prediction')
    return np.maximum(q,1e-12)

def metric_sums(y,p,s):
    y=np.asarray(y);p=np.asarray(p);ly=np.log(y);lp=np.log(p)
    return dict(n=len(y),sse_q=float(np.sum((y-p)**2)),sse_ln=float(np.sum((ly-lp)**2)),
                sae_q=float(np.sum(np.abs(y-p))),sae_ln=float(np.sum(np.abs(ly-lp))),
                sum_y=float(y.sum()),sum_y2=float((y*y).sum()),sum_lny=float(ly.sum()),
                sum_lny2=float((ly*ly).sum()),n_violation=int(np.sum((p<0)|(p>s*(1+1e-8)))))

def derive(z):
    z=dict(z);n=float(z['n'])
    z.update(rmse=float(np.sqrt(z['sse_q']/n)),ln_rmse=float(np.sqrt(z['sse_ln']/n)),
             mae=float(z['sae_q']/n),ln_mae=float(z['sae_ln']/n),
             r2=float(1-z['sse_q']/max(z['sum_y2']-z['sum_y']**2/n,1e-30)),
             ln_r2=float(1-z['sse_ln']/max(z['sum_lny2']-z['sum_lny']**2/n,1e-30)),
             violation_rate=float(z['n_violation']/n))
    return z

def paired_material_bootstrap(candidate,comparator,n_boot=4000,seed=20261025):
    """Rows carry material labels and additive scene statistics, including repeats."""
    a=candidate.groupby('target_material')[ADDITIVE].sum().sort_index()
    b=comparator.groupby('target_material')[ADDITIVE].sum().reindex(a.index)
    if b.isna().any().any() or not np.array_equal(a.n,b.n):
        raise ValueError('Both configurations must contain identical material/query support')
    rng=np.random.default_rng(seed);m=len(a)
    counts=np.asarray([np.bincount(rng.integers(m,size=m),minlength=m) for _ in range(n_boot)])
    out={}
    for metric,sse in [('rmse','sse_q'),('ln_rmse','sse_ln')]:
        n=a.n.to_numpy();sa=a[sse].to_numpy();sb=b[sse].to_numpy()
        draws=np.sqrt(counts@sa/(counts@n))-np.sqrt(counts@sb/(counts@n))
        out[metric]={'difference':float(np.sqrt(sa.sum()/n.sum())-np.sqrt(sb.sum()/n.sum())),
                     'interval':np.quantile(draws,[.025,.975]).tolist()}
    return out
