"""Source-only empirical priors, calibration and finite-budget active design.

Gaussian updates, IVR, kernels and robust IRLS are established ingredients.
"""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import Ridge
from numpy.polynomial.hermite import hermgauss

EPS=1e-5
GH_X,GH_W=hermgauss(12)
GH_W=GH_W/np.sqrt(np.pi)

def conditions(x):
    return np.c_[np.log(x.Ci),np.log(x.dose_calculated),np.log(x['Adsorption_time (min)']),x['solution pH']]

def fraction(q,s):return np.clip(np.asarray(q)/np.asarray(s),EPS,1-EPS)

def latent_observations(y,q,s,space):
    if space=='ln':return np.log(y)-np.log(q)
    return logit(fraction(y,s))-logit(fraction(q,s))

def robust_fit(H,z,w):
    w0=np.asarray(w).copy();w=w0.copy();b=np.zeros(H.shape[1])
    for _ in range(6):
        b=np.linalg.solve(H.T@(w[:,None]*H)+.02*np.eye(H.shape[1]),H.T@(w*z))
        e=z-H@b;mad=max(1.4826*np.median(np.abs(e-np.median(e))),.1)
        w=w0*np.minimum(1.,2.5*mad/np.maximum(np.abs(e),1e-8))
    return b

class EmpiricalPrior:
    """Source material OOF residuals only; source labels are allowed auxiliary data."""
    def __init__(self,source,q0,space):
        self.space=space
        s=source.available_mass_per_g.to_numpy();y=source.qe.to_numpy();q0=np.asarray(q0)
        self.sm=float(np.mean(np.log(s)));self.ss=max(float(np.std(np.log(s))),.25)
        zc=conditions(source);self.xm=zc.mean(0);self.xs=np.maximum(zc.std(0),.25)
        H=self.phi(source);z=latent_observations(y,q0,s,space)
        r0=fraction(q0,s);lnjac=1-r0 if space=='logit' else np.ones(len(source))
        betas=[];noise=[];local=[]
        groups=source.material_group.to_numpy()
        for group in sorted(np.unique(groups)):
            ix=np.flatnonzero(groups==group)
            w=np.maximum(lnjac[ix],.05)**2 if space=='logit' else np.ones(len(ix))
            beta=robust_fit(H[ix],z[ix],w);betas.append(beta)
            err=z[ix]-H[ix]@beta
            # Working residual dispersion, not measured instrument noise.
            scaled=err*lnjac[ix]
            sigma=max(1.4826*np.median(np.abs(scaled-np.median(scaled))),.15)
            noise.append(sigma**2)
            local.append(np.var(np.clip(err,np.quantile(err,.1),np.quantile(err,.9))))
        b=np.asarray(betas);self.mu=np.median(b,axis=0)
        bb=np.clip(b,np.quantile(b,.1,axis=0),np.quantile(b,.9,axis=0))
        cov=np.cov(bb.T)+.04*np.eye(2)
        self.cov=.8*cov+.2*np.diag(np.diag(cov))
        self.noise=max(float(np.median(noise)),.15**2)
        self.local=max(float(np.median(local))*.5,.04)
        self.global_var=max(float(np.var(np.clip(z,np.quantile(z,.05),np.quantile(z,.95)))),.1)
        self.vq=max(float(np.var(y)),1e-6);self.vl=max(float(np.var(np.log(y))),1e-6)
        self.spec={'space':space,'source_material_count':len(b),'mean':self.mu.tolist(),'covariance':self.cov.tolist(),
            'residual_noise':self.noise,'local_amplitude':self.local,'global_variance':self.global_var,
            'length_scale':1.5,'working_noise_not_experimental_measurement_error':True,
            'hyperparameter_selection':'fixed robust moment rules; no target query labels'}

    def phi(self,x):return np.c_[np.ones(len(x)),(np.log(x.available_mass_per_g)-self.sm)/self.ss]
    def zx(self,x):return (conditions(x)-self.xm)/self.xs
    def mean(self,x,kind):return np.zeros(len(x)) if kind=='gp' else self.phi(x)@self.mu
    def kernel(self,a,b,kind='linear'):
        if kind=='gp':
            za=self.zx(a);zb=self.zx(b)
            return self.global_var*np.exp(-.5*np.sum((za[:,None,:]-zb[None,:,:])**2,axis=2)/(1.5**2))
        K=self.phi(a)@self.cov@self.phi(b).T
        if kind=='hybrid':
            za=self.zx(a);zb=self.zx(b)
            K+=self.local*np.exp(-.5*np.sum((za[:,None,:]-zb[None,:,:])**2,axis=2)/(1.5**2))
        return K
    def diag(self,x,kind='linear'):
        if kind=='gp':return np.full(len(x),self.global_var)
        v=np.einsum('ij,jk,ik->i',self.phi(x),self.cov,self.phi(x))
        return v+(self.local if kind=='hybrid' else 0)
    def observation_noise(self,q0,s,hetero):
        r0=fraction(q0,s)
        if hetero and self.space=='logit':return self.noise/np.maximum(1-r0,.05)**2
        return np.full(len(q0),self.noise)

def posterior(prior,x,q0,anchors,qa,ya,kind='linear',hetero=False,robust=False):
    mean=prior.mean(x,kind);var=prior.diag(x,kind)
    if len(anchors)==0:return mean,var
    Kaa=prior.kernel(anchors,anchors,kind);Kqa=prior.kernel(x,anchors,kind)
    sa=anchors.available_mass_per_g.to_numpy()
    z=latent_observations(ya,qa,sa,prior.space);ma=prior.mean(anchors,kind)
    noise=prior.observation_noise(qa,sa,hetero)
    w=np.ones(len(anchors))
    for _ in range(6 if robust else 1):
        B=Kaa+np.diag(noise/w)+1e-8*np.eye(len(anchors))
        alpha=np.linalg.solve(B,z-ma)
        if robust:
            e=(z-ma)-Kaa@alpha
            w=np.clip(5/(4+e**2/np.maximum(noise,1e-8)),.05,1)
    mean=mean+Kqa@alpha
    var=np.maximum(var-np.sum(Kqa*np.linalg.solve(B,Kqa.T).T,axis=1),1e-9)
    return mean,var

def inverse_prediction(q0,s,mean,var,space,readout='plugin'):
    q0=np.asarray(q0);s=np.asarray(s)
    if space=='ln':return np.clip(q0*np.exp(np.clip(mean,-20,20)),1e-12,s)
    f0=logit(fraction(q0,s))
    if readout=='mean':
        latent=f0[:,None]+mean[:,None]+np.sqrt(2*var[:,None])*GH_X
        return s*np.sum(expit(latent)*GH_W,axis=1)
    return s*expit(f0+mean)

CALIBRATORS=['Source_bound','Ln_offset_bound','Ln_ridge_bound','Logit_ridge',
    'EB_ln','GP_ln','EB_ln_kernel','EB_logit_homo','EB_logit_hetero',
    'EB_logit_kernel','EB_logit_robust','EB_logit_posterior_mean']

def calibrate(method,x,q0,anchors,qa,ya,priors):
    s=x.available_mass_per_g.to_numpy()
    if method=='Source_bound' or (len(anchors)==0 and method in ['Ln_offset_bound','Ln_ridge_bound','Logit_ridge']):return np.clip(q0,1e-12,s)
    sa=anchors.available_mass_per_g.to_numpy()
    if method.startswith('Ln_'):
        delta=np.log(ya)-np.log(qa)
        if method=='Ln_offset_bound':corr=np.full(len(x),delta.mean())
        else:corr=Ridge(alpha=4).fit(np.log(sa)[:,None],delta).predict(np.log(s)[:,None])
        return np.clip(q0*np.exp(np.clip(corr,-20,20)),1e-12,s)
    if method=='Logit_ridge':
        delta=latent_observations(ya,qa,sa,'logit')
        corr=Ridge(alpha=4).fit(np.log(sa)[:,None],delta).predict(np.log(s)[:,None])
        return inverse_prediction(q0,s,corr,np.zeros(len(x)),'logit')
    space='ln' if method in ['EB_ln','GP_ln','EB_ln_kernel'] else 'logit'
    kind='gp' if method=='GP_ln' else ('hybrid' if method in ['EB_ln_kernel','EB_logit_kernel','EB_logit_robust','EB_logit_posterior_mean'] else 'linear')
    hetero=space=='logit' and method!='EB_logit_homo'
    robust=method=='EB_logit_robust'
    mean,var=posterior(priors[space],x,q0,anchors,qa,ya,kind,hetero,robust)
    return inverse_prediction(q0,s,mean,var,space,'mean' if method=='EB_logit_posterior_mean' else 'plugin')

class BudgetedLabels:
    def __init__(self,ids,blocks,labels,max_k=5):
        self.ids=np.asarray(ids);self.blocks=np.asarray(blocks);self._labels=np.asarray(labels)
        self.max_k=max_k;self.access=[]
    def reveal(self,j):
        if len(self.access)>=self.max_k:raise AssertionError('label budget exceeded')
        if j in self.access:raise AssertionError('duplicate anchor')
        if self.blocks[j] in set(self.blocks[self.access]):raise AssertionError('same condition block twice')
        self.access.append(int(j));return float(self._labels[j])

def select_anchors(policy,candidates,qc,query,qq,priors,oracle,seed,max_k=5):
    """Sequential policies; labels revealed only after selecting the current row."""
    rng=np.random.default_rng(seed);blocks=candidates.condition_block.to_numpy();chosen=[];ys=[];records=[]
    active=policy.startswith('IVR')
    space='ln' if policy=='IVR_ln' else 'logit'
    prior=priors[space];kind='hybrid' if policy=='IVR_link_kernel' else 'linear'
    if active:
        Kcc=prior.kernel(candidates,candidates,kind);Kqc=prior.kernel(query,candidates,kind)
        Kqqdiag=prior.diag(query,kind)
    Z=conditions(candidates);Z=(Z-Z.mean(0))/np.maximum(Z.std(0),.25)
    # All query blocks contribute equally to integrated risk.
    nq=query.groupby('condition_block').excel_row.transform('size').to_numpy()
    qw=1/nq;qw=qw/qw.sum()
    for step in range(max_k):
        eligible=np.array([j for j,b in enumerate(blocks) if b not in set(blocks[chosen])])
        if policy.startswith('random'):
            j=int(rng.choice(eligible));gain=np.nan
        elif policy=='maximin':
            dist=np.sum(Z[eligible]**2,axis=1) if not chosen else np.min(np.sum((Z[eligible,None]-Z[chosen][None])**2,axis=2),axis=1)
            j=int(eligible[np.argmin(dist) if not chosen else np.argmax(dist)]);gain=np.nan
        elif policy=='D_opt':
            H=priors['ln'].phi(candidates)
            precision=np.eye(2)+H[chosen].T@H[chosen] if chosen else np.eye(2)
            iv=np.linalg.inv(precision);score=np.einsum('ij,jk,ik->i',H,iv,H)
            j=int(eligible[np.argmax(score[eligible])]);gain=float(score[j])
        else:
            if chosen:
                aa=candidates.iloc[chosen];qa=qc[chosen]
                noise=prior.observation_noise(qa,aa.available_mass_per_g.to_numpy(),space=='logit')
                B=Kcc[np.ix_(chosen,chosen)]+np.diag(noise)+1e-8*np.eye(len(chosen))
                qcov=Kqc-Kqc[:,chosen]@np.linalg.solve(B,Kcc[chosen])
                cv=np.maximum(np.diag(Kcc)-np.sum(Kcc[:,chosen]*np.linalg.solve(B,Kcc[chosen]).T,axis=1),1e-9)
                pm,_=posterior(prior,query,qq,aa,qa,np.asarray(ys),kind,space=='logit')
                pcm,_=posterior(prior,candidates,qc,aa,qa,np.asarray(ys),kind,space=='logit')
            else:
                qcov=Kqc;cv=np.diag(Kcc);pm=prior.mean(query,kind);pcm=prior.mean(candidates,kind)
            if space=='logit':
                sq=query.available_mass_per_g.to_numpy();pred=inverse_prediction(qq,sq,pm,np.zeros(len(query)),'logit')
                rq=fraction(pred,sq);jl=1-rq;jq=pred*jl
                weights=qw*.5*(jq**2/prior.vq+jl**2/prior.vl)
                sc=candidates.available_mass_per_g.to_numpy();pc=inverse_prediction(qc,sc,pcm,np.zeros(len(candidates)),'logit')
                rn=prior.observation_noise(pc,sc,True)
            else:
                weights=qw;rn=prior.observation_noise(qc,candidates.available_mass_per_g.to_numpy(),False)
            scores=np.sum(weights[:,None]*qcov**2,axis=0)/(cv+rn)
            j=int(eligible[np.argmax(scores[eligible])]);gain=float(scores[j])
        y=oracle.reveal(j);chosen.append(j);ys.append(y)
        records.append({'step':step+1,'excel_row':int(candidates.excel_row.iloc[j]),'condition_block':int(blocks[j]),'predicted_gain':gain})
    assert chosen==oracle.access
    return chosen,np.asarray(ys),records
