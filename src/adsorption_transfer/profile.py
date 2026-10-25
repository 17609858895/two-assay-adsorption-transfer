"""Anchor-profiled logit discrepancy calibration; ordinary kriging adaptation.

The free target intercept enforces translation equivariance in observed latent
discrepancies; source OOF information supplies contrast/condition structure.
This is an adaptation of universal kriging, not a new general kriging method.
"""
import numpy as np
from .methods import latent_observations,inverse_prediction

PROFILE_METHODS=['Profile_hybrid','Profile_linear','Profile_zero_mean','Weighted_logit_offset']

def profile_predict(method,x,q0,anchors,qa,ya,prior,return_details=False):
    s=x.available_mass_per_g.to_numpy();sa=anchors.available_mass_per_g.to_numpy()
    if not len(anchors):
        pred=np.clip(q0,1e-12,s)
        return (pred,{'n_anchors':0}) if return_details else pred
    z=latent_observations(ya,qa,sa,'logit')
    noise=prior.observation_noise(qa,sa,True)
    if method=='Weighted_logit_offset':
        w=1/noise;w=w/w.sum();mean=np.full(len(x),w@z);var=np.full(len(x),1/np.sum(1/noise))
        W=np.tile(w,(len(x),1));offset=float(w@z)
    else:
        assert method in PROFILE_METHODS
        kind='linear' if method=='Profile_linear' else 'hybrid'
        ma=prior.mean(anchors,kind);mq=prior.mean(x,kind)
        if method=='Profile_zero_mean':ma=np.zeros(len(anchors));mq=np.zeros(len(x))
        K=prior.kernel(anchors,anchors,kind);C=prior.kernel(x,anchors,kind)
        B=K+np.diag(noise)+1e-8*np.eye(len(anchors))
        one=np.ones(len(anchors));bi1=np.linalg.solve(B,one);den=float(one@bi1)
        offset=float(bi1@(z-ma)/den)
        mean=mq+offset+C@np.linalg.solve(B,z-ma-offset)
        W=np.linalg.solve(B,C.T).T+(1-C@bi1)[:,None]*(bi1/den)[None,:]
        var=prior.diag(x,kind)-np.sum(C*np.linalg.solve(B,C.T).T,axis=1)+(1-C@bi1)**2/den
    assert np.max(np.abs(W.sum(1)-1))<1e-8
    var=np.maximum(var,1e-10)
    pred=inverse_prediction(q0,s,mean,var,'logit')
    details={'n_anchors':len(anchors),'profiled_intercept':offset,'minimum_latent_variance':float(var.min()),
        'maximum_weight_sum_error':float(np.max(np.abs(W.sum(1)-1))),
        'maximum_query_l1_leverage':float(np.max(np.sum(np.abs(W),axis=1))),
        'mean_query_l1_leverage':float(np.mean(np.sum(np.abs(W),axis=1))),
        'anchor_log_mass_range':float(np.ptp(np.log(sa))),'maximum_absolute_anchor_residual':float(np.max(np.abs(z)))}
    return (pred,details) if return_details else pred
