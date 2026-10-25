"""Source-task residual mixtures for budgeted, bounded calibration (v4).

Frozen adaptation candidate; mixture modelling and Gaussian conditioning are
established.  The testable change retains source-material residual modes and
lets only revealed anchors identify those modes, versus moment compression.
No target-query labels or unselected candidate labels enter this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit, logsumexp
from .methods import EmpiricalPrior, robust_fit, fraction, latent_observations

METHODS=['SCM_uniform','SCM_evidence','SCM_moment',
    'SCM_contrast_uniform','SCM_contrast_evidence','SCM_contrast_moment']

@dataclass
class SourceMixturePosterior:
    means: np.ndarray             # query x source-material components
    within_variance: np.ndarray
    weights: np.ndarray
    log_evidence: np.ndarray

    @property
    def latent_mean(self):
        return self.means @ self.weights

    @property
    def latent_variance(self):
        return self.within_variance + (self.means**2)@self.weights-self.latent_mean**2

    @property
    def effective_components(self):
        return float(1/np.sum(self.weights**2))

class SourceMixturePrior:
    """Equal source-task prior, common within-task kernel, anchor Bayes weights.

    Source tasks use the same robust OOF residual regression as v2, without
    clipping coefficients into a unimodal between-task covariance. The .04
    within-task coefficient covariance is v2's fixed regularization floor.
    The matched Gaussian has exactly the mixture's first two prior moments.
    """
    def __init__(self, source, q0, global_prior=None):
        self.global_prior=global_prior or EmpiricalPrior(source,q0,'logit')
        self.space='logit';self.groups=sorted(source.material_group.unique())
        assert len(self.groups)>=2
        H=self.phi(source);s=source.available_mass_per_g.to_numpy()
        r=fraction(q0,s);z=latent_observations(source.qe.to_numpy(),q0,s,'logit')
        groups=source.material_group.to_numpy();betas=[];sizes=[];expert_diagnostics=[]
        for group in self.groups:
            ix=np.flatnonzero(groups==group)
            beta=robust_fit(H[ix],z[ix],np.maximum(1-r[ix],.05)**2)
            betas.append(beta)
            sizes.append(int(len(ix)))
            error=z[ix]-H[ix]@beta
            scaled=error*(1-r[ix])
            noise=max(1.4826*np.median(np.abs(scaled-np.median(scaled))),.15)**2
            mad=max(1.4826*np.median(np.abs(error-np.median(error))),.1)
            w=np.maximum(1-r[ix],.05)**2*np.minimum(1.,2.5*mad/np.maximum(np.abs(error),1e-8))
            expert_diagnostics.append(dict(material_group=group,n_rows=int(len(ix)),
                n_blocks=int(source.iloc[ix].condition_block.nunique()),beta=beta.tolist(),
                working_noise=float(noise),effective_rows=float(w.sum()**2/np.sum(w*w)),
                working_noise_used='common robust-median source noise in frozen candidate'))
        self.betas=np.asarray(betas);self.mu=self.betas.mean(0)
        centered=self.betas-self.mu
        self.between_cov=centered.T@centered/len(self.groups)
        self.within_cov=.04*np.eye(2)
        self.cov=self.within_cov+self.between_cov
        self.noise=self.global_prior.noise;self.local=self.global_prior.local
        self.vq=self.global_prior.vq;self.vl=self.global_prior.vl
        self.spec=dict(candidate='source-material residual mixture',space='logit',
            source_groups=self.groups,source_group_sizes=sizes,betas=self.betas.tolist(),experts=expert_diagnostics,
            component_weights='uniform before anchors; likelihood weights for SCM_evidence',
            component_mean='h(x) beta_g; v2 source-material OOF robust_fit',
            within_coefficient_covariance=self.within_cov.tolist(),
            moment_mean=self.mu.tolist(),between_covariance=self.between_cov.tolist(),
            local_amplitude=self.local,noise=self.noise,length_scale=1.5,
            readout='source logit plus posterior mean residual; same plug-in as v2',
            target_query_labels_used=False,hyperparameter_tuning='none; frozen before scoring')

    def phi(self,x):return self.global_prior.phi(x)
    def zx(self,x):return self.global_prior.zx(x)
    def means(self,x):return self.phi(x)@self.betas.T
    def kernel(self,a,b,moment=False):
        za=self.zx(a);zb=self.zx(b)
        cov=self.cov if moment else self.within_cov
        return self.phi(a)@cov@self.phi(b).T+self.local*np.exp(-.5*np.sum((za[:,None,:]-zb[None,:,:])**2,axis=2)/(1.5**2))
    def diag(self,x,moment=False):
        cov=self.cov if moment else self.within_cov
        return np.einsum('ij,jk,ik->i',self.phi(x),cov,self.phi(x))+self.local
    def observation_noise(self,q0,s,hetero=True):
        return self.global_prior.observation_noise(q0,s,hetero)

    def posterior(self,x,q0,anchors,qa,ya,weighting='evidence'):
        m=self.means(x);v=self.diag(x);G=len(self.groups)
        log_evidence=np.zeros(G)
        if len(anchors):
            sa=anchors.available_mass_per_g.to_numpy()
            za=latent_observations(ya,qa,sa,'logit')
            ma=self.means(anchors)
            B=self.kernel(anchors,anchors)+np.diag(self.observation_noise(qa,sa))+1e-8*np.eye(len(anchors))
            L=np.linalg.cholesky(B)
            error=za[:,None]-ma
            alpha=np.linalg.solve(B,error)
            Kqa=self.kernel(x,anchors)
            m=m+Kqa@alpha
            v=np.maximum(v-np.sum(Kqa*np.linalg.solve(B,Kqa.T).T,axis=1),1e-9)
            whitened=np.linalg.solve(L,error)
            log_evidence=-.5*np.sum(whitened**2,axis=0)-np.log(np.diag(L)).sum()-.5*len(anchors)*np.log(2*np.pi)
        weights=np.full(G,1/G) if weighting=='uniform' else np.exp(log_evidence-logsumexp(log_evidence))
        return SourceMixturePosterior(m,v,weights,log_evidence)

    def moment_posterior(self,x,q0,anchors,qa,ya):
        m=self.phi(x)@self.mu;v=self.diag(x,moment=True)
        if len(anchors):
            sa=anchors.available_mass_per_g.to_numpy()
            z=latent_observations(ya,qa,sa,'logit')
            B=self.kernel(anchors,anchors,moment=True)+np.diag(self.observation_noise(qa,sa))+1e-8*np.eye(len(anchors))
            Kqa=self.kernel(x,anchors,moment=True)
            m=m+Kqa@np.linalg.solve(B,z-self.phi(anchors)@self.mu)
            v=np.maximum(v-np.sum(Kqa*np.linalg.solve(B,Kqa.T).T,axis=1),1e-9)
        return m,v

    def contrast_posterior(self,x,q0,anchors,qa,ya,weighting='evidence',moment=False):
        """Profile an unrestricted common nuisance offset within each expert.

        Restricted likelihood compares only identifiable anchor contrasts.
        Common B makes determinant terms common to all mixture components.
        Universal-kriging variance includes uncertainty in the free offset.
        With zero anchors that offset is unidentified; the public predictor
        returns the frozen bounded source rather than pretending a proper prior.
        """
        if not len(anchors):raise ValueError('free offset requires at least one anchor')
        s=anchors.available_mass_per_g.to_numpy()
        z=latent_observations(ya,qa,s,'logit')
        ma=(self.phi(anchors)@self.mu)[:,None] if moment else self.means(anchors)
        mq=(self.phi(x)@self.mu)[:,None] if moment else self.means(x)
        B=self.kernel(anchors,anchors,moment=moment)+np.diag(self.observation_noise(qa,s))+1e-8*np.eye(len(anchors))
        one=np.ones(len(anchors));inv_one=np.linalg.solve(B,one);den=float(one@inv_one)
        error=z[:,None]-ma
        offset=(inv_one@error)/den
        centered=error-one[:,None]*offset[None,:]
        Kqa=self.kernel(x,anchors,moment=moment)
        means=mq+offset[None,:]+Kqa@np.linalg.solve(B,centered)
        v=self.diag(x,moment=moment)-np.sum(Kqa*np.linalg.solve(B,Kqa.T).T,axis=1)
        c=1-Kqa@inv_one
        v=np.maximum(v+c*c/den,1e-9)
        logdet=2*np.log(np.diag(np.linalg.cholesky(B))).sum()
        log_evidence=-.5*np.sum(centered*np.linalg.solve(B,centered),axis=0)-.5*logdet-.5*np.log(den)-.5*(len(anchors)-1)*np.log(2*np.pi)
        G=means.shape[1]
        weights=np.full(G,1/G) if weighting=='uniform' else np.exp(log_evidence-logsumexp(log_evidence))
        return SourceMixturePosterior(means,v,weights,log_evidence),offset

def mixture_calibrate(method,x,q0,anchors,qa,ya,prior,return_diagnostics=False):
    if method not in METHODS:raise ValueError(method)
    if method.startswith('SCM_contrast'):
        if not len(anchors):
            s=x.available_mass_per_g.to_numpy();pred=np.clip(q0,1e-12,s)
            diag=dict(effective_components=np.nan,max_weight=np.nan,weights=[],log_evidence=[],
                free_offset_unidentified=True,zero_assay_action='bounded frozen source',bounded=True)
            return (pred,diag) if return_diagnostics else pred
        post,offset=prior.contrast_posterior(x,q0,anchors,qa,ya,
            weighting='uniform' if method=='SCM_contrast_uniform' else 'evidence',
            moment=method=='SCM_contrast_moment')
        mean,var=post.latent_mean,post.latent_variance
        diag=dict(effective_components=post.effective_components,max_weight=float(post.weights.max()),
            weights=post.weights.tolist(),log_evidence=post.log_evidence.tolist(),profile_offsets=offset.tolist(),
            free_offset_unidentified=False)
    elif method=='SCM_moment':
        mean,var=prior.moment_posterior(x,q0,anchors,qa,ya)
        diag=dict(effective_components=np.nan,max_weight=np.nan,weights=[],log_evidence=[])
    else:
        post=prior.posterior(x,q0,anchors,qa,ya,'uniform' if method=='SCM_uniform' else 'evidence')
        mean,var=post.latent_mean,post.latent_variance
        diag=dict(effective_components=post.effective_components,max_weight=float(post.weights.max()),
            weights=post.weights.tolist(),log_evidence=post.log_evidence.tolist())
    s=x.available_mass_per_g.to_numpy()
    pred=s*expit(logit(fraction(q0,s))+mean)
    diag.update(mean_latent_variance=float(np.mean(var)),bounded=bool(np.all((pred>0)&(pred<=s))))
    return (pred,diag) if return_diagnostics else pred
