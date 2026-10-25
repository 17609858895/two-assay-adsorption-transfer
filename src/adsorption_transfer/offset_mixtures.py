"""Proper narrow/broad source-offset mixture for few-assay calibration.

Robust mixture priors and Bayesian model averaging are established. This frozen
task adaptation changes only uncertainty in a new-material global intercept.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from .methods import latent_observations,inverse_prediction

ROBUST_OFFSET_METHODS=['Robust_offset_fixed_half','Robust_offset_evidence','Robust_offset_moment']

@dataclass
class RobustOffsetPosterior:
    means:np.ndarray
    variances:np.ndarray
    weights:np.ndarray
    log_evidence:np.ndarray
    broad_offset_variance:float

    @property
    def latent_mean(self):return self.means@self.weights
    @property
    def latent_variance(self):
        return np.maximum((self.variances+self.means*self.means)@self.weights-self.latent_mean**2,1e-9)

def component_posterior(x,q0,anchors,qa,ya,prior,extra_variance=0.):
    mean=prior.mean(x,'hybrid');variance=prior.diag(x,'hybrid')+extra_variance
    log_evidence=0.
    if len(anchors):
        sa=anchors.available_mass_per_g.to_numpy()
        z=latent_observations(ya,qa,sa,'logit');ma=prior.mean(anchors,'hybrid')
        B=prior.kernel(anchors,anchors,'hybrid')+extra_variance+np.diag(prior.observation_noise(qa,sa,True))+1e-8*np.eye(len(anchors))
        C=prior.kernel(x,anchors,'hybrid')+extra_variance
        error=z-ma;alpha=np.linalg.solve(B,error)
        mean=mean+C@alpha
        variance=np.maximum(variance-np.sum(C*np.linalg.solve(B,C.T).T,axis=1),1e-9)
        L=np.linalg.cholesky(B);whitened=np.linalg.solve(L,error)
        log_evidence=float(-.5*(whitened@whitened)-np.log(np.diag(L)).sum()-.5*len(anchors)*np.log(2*np.pi))
    return mean,variance,log_evidence

def robust_offset_posterior(x,q0,anchors,qa,ya,prior,evidence=True):
    vb=9*float(prior.cov[0,0]);assert vb>0
    components=[component_posterior(x,q0,anchors,qa,ya,prior,v) for v in [0.,vb]]
    means=np.column_stack([p[0] for p in components]);variances=np.column_stack([p[1] for p in components])
    log_evidence=np.array([p[2] for p in components])
    weights=np.exp(log_evidence-logsumexp(log_evidence)) if evidence else np.full(2,.5)
    return RobustOffsetPosterior(means,variances,weights,log_evidence,vb)

def robust_offset_predict(method,x,q0,anchors,qa,ya,prior,return_details=False):
    assert method in ROBUST_OFFSET_METHODS
    vb=9*float(prior.cov[0,0])
    if method=='Robust_offset_moment':
        mean,var,log_evidence=component_posterior(x,q0,anchors,qa,ya,prior,.5*vb)
        detail=dict(component_weights=[],component_log_evidence=[log_evidence],broad_offset_variance=vb,
            matched_gaussian_extra_variance=.5*vb,effective_components=1.)
    else:
        posterior=robust_offset_posterior(x,q0,anchors,qa,ya,prior,evidence=method=='Robust_offset_evidence')
        mean,var=posterior.latent_mean,posterior.latent_variance
        detail=dict(component_weights=posterior.weights.tolist(),component_log_evidence=posterior.log_evidence.tolist(),
            broad_offset_variance=vb,broad_component_probability=float(posterior.weights[1]),
            effective_components=float(1/np.sum(posterior.weights**2)))
    s=x.available_mass_per_g.to_numpy()
    prediction=inverse_prediction(q0,s,mean,var,'logit')
    assert np.isfinite(prediction).all() and (prediction>0).all() and (prediction<=s*(1+1e-8)).all()
    detail.update(n_anchors=len(anchors),minimum_latent_variance=float(var.min()),
        mean_latent_variance=float(var.mean()),maximum_absolute_latent_mean=float(np.max(np.abs(mean))),bounded=True)
    return (prediction,detail) if return_details else prediction
