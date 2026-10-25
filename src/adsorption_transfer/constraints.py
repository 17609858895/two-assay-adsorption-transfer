"""Nonnegative, constant-preserving kriging weights for few-assay calibration.

Constrained kriging is established. The task-specific candidate uses its
maximum-principle/noise-amplification properties for bounded logit residuals.
"""
from itertools import combinations
import numpy as np
from .methods import latent_observations,inverse_prediction

CONVEX_METHODS=['Convex_contrast','Convex_source_mean']

def convex_weights(B,C):
    """Exact active-set enumeration for k<=5, on each independent query row."""
    n,k=C.shape;best=np.full(n,np.inf);W=np.zeros((n,k))
    assert k<=5
    for size in range(1,k+1):
        for subset in combinations(range(k),size):
            ids=np.array(subset);sub=B[np.ix_(ids,ids)];cross=C[:,ids]
            bi1=np.linalg.solve(sub,np.ones(size));den=bi1.sum()
            w=np.linalg.solve(sub,cross.T).T+(1-cross@bi1)[:,None]*(bi1/den)[None,:]
            ok=np.all(w>=-1e-10,axis=1)
            objective=np.einsum('ij,jk,ik->i',w,sub,w)-2*np.sum(w*cross,axis=1)
            take=ok&(objective<best-1e-12)
            W[take]=0;rows=np.flatnonzero(take)
            W[np.ix_(rows,ids)]=w[take];best[take]=objective[take]
    assert np.isfinite(best).all()
    W=np.maximum(W,0);W=W/W.sum(1)[:,None]
    assert W.min()>=0 and np.max(np.abs(W.sum(1)-1))<1e-12
    return W

def convex_predict(method,x,q0,anchors,qa,ya,prior,return_details=False):
    assert method in CONVEX_METHODS
    s=x.available_mass_per_g.to_numpy();sa=anchors.available_mass_per_g.to_numpy()
    if not len(anchors):
        p=np.clip(q0,1e-12,s);return (p,{'n_anchors':0}) if return_details else p
    K=prior.kernel(anchors,anchors,'hybrid');C=prior.kernel(x,anchors,'hybrid')
    B=K+np.diag(prior.observation_noise(qa,sa,True))+1e-8*np.eye(len(anchors))
    W=convex_weights(B,C)
    z=latent_observations(ya,qa,sa,'logit')
    mq=prior.mean(x,'hybrid');ma=prior.mean(anchors,'hybrid')
    if method=='Convex_contrast':mq=np.zeros(len(x));ma=np.zeros(len(anchors))
    mean=mq+W@(z-ma)
    pred=np.clip(inverse_prediction(q0,s,mean,np.zeros(len(x)),'logit'),1e-12,s)
    detail={'n_anchors':len(anchors),'minimum_influence_weight':float(W.min()),
        'maximum_query_l1_leverage':float(np.max(np.sum(np.abs(W),axis=1))),
        'maximum_weight_sum_error':float(np.max(np.abs(W.sum(1)-1))),
        'fraction_query_on_simplex_boundary':float(np.mean(np.any(W<1e-10,axis=1))),
        'anchor_log_mass_range':float(np.ptp(np.log(sa)))}
    return (pred,detail) if return_details else pred
