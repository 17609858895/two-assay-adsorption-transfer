"""Exhaustive two-assay design using actual-estimator working risks."""
import numpy as np
from .methods import fraction
from .constraints import convex_weights
POLICIES=['Cross_estimator_minimax_pair','EB_minimax_pair','Cross_estimator_average_pair']
BATCH_POLICIES=['simple_convex_minimax','simple_convex_average','simple_profile_minimax',
                'simple_bayes_minimax','bayes_minimax','simple_bayes_average']

def forbidden_guard(x):
    bad=set(x.columns)&{'qe','Cf','removal_fraction'}
    if bad:raise AssertionError('Design received outcome columns: '+repr(bad))

def pair_scores(candidates,qc,query,qq,prior):
    """Return every finite-pool pair's four normalized working risks.

    Risk includes query working noise Rq, since the target is a future observed
    residual.  Rq is constant across pairs but stays in the normalization.  The
    zero-correction reference is m_q**2 + diag(Kqq) + Rq.  Both inverse-link
    Jacobians are evaluated at the bounded source predictor, not the prior mean.
    """
    forbidden_guard(candidates);forbidden_guard(query)
    blocks=candidates.condition_block.to_numpy()
    ii,jj=np.triu_indices(len(candidates),1)
    ok=blocks[ii]!=blocks[jj];ii=ii[ok];jj=jj[ok]
    assert len(ii)>0
    sq=query.available_mass_per_g.to_numpy();sc=candidates.available_mass_per_g.to_numpy()
    Kcc=prior.kernel(candidates,candidates,'hybrid')
    Kqc=prior.kernel(query,candidates,'hybrid')
    mq=prior.mean(query,'hybrid');mc=prior.mean(candidates,'hybrid')
    noise=prior.observation_noise(qc,sc,True)
    vn=prior.diag(query,'hybrid')+prior.observation_noise(qq,sq,True)
    nq=query.groupby('condition_block').excel_row.transform('size').to_numpy()
    qw=(1/nq);qw=qw/qw.sum()
    r=fraction(qq,sq);qref=sq*r
    weights=np.c_[qw*(qref*(1-r))**2,qw*(1-r)**2]
    reference=np.sum(weights*(vn+mq**2)[:,None],axis=0)
    assert np.all(reference>0)
    xc=np.log(sc);xq=np.log(sq)
    allrisk=np.empty((len(ii),4))
    for left in range(0,len(ii),2048):
        sl=slice(left,left+2048);i=ii[sl];j=jj[sl]
        # Exact Ridge(alpha=4, fit_intercept=True) linear smoother for two rows.
        xm=(xc[i]+xc[j])/2;dx=(xc[i]-xc[j])/2
        l1=.5+(xq[:,None]-xm[None,:])*dx[None,:]/(4+2*dx**2)[None,:]
        l2=1-l1
        a=Kcc[i,i]+noise[i];b=Kcc[j,j]+noise[j];c=Kcc[i,j]
        u=Kqc[:,i];v=Kqc[:,j]
        bias=mq[:,None]-l1*mc[i][None,:]-l2*mc[j][None,:]
        sr=(bias**2+vn[:,None]-2*(l1*u+l2*v)+
            l1*l1*a[None,:]+2*l1*l2*c[None,:]+l2*l2*b[None,:])
        # Same 1e-8 jitter as existing EB_logit_kernel posterior implementation.
        aa=a+1e-8;bb=b+1e-8;det=aa*bb-c*c
        assert np.all(det>0)
        er=vn[:,None]-(bb[None,:]*u*u-2*c[None,:]*u*v+aa[None,:]*v*v)/det[None,:]
        assert sr.min()>-1e-7 and er.min()>-1e-7
        allrisk[sl,:2]=(weights.T@np.maximum(sr,0)/reference[:,None]).T
        allrisk[sl,2:]=(weights.T@np.maximum(er,0)/reference[:,None]).T
    objectives=np.c_[allrisk.max(1),allrisk[:,2:].max(1),allrisk.mean(1)]
    # Lexical row ordering resolves exact numerical ties without target outcomes.
    choices={p:int(np.argmin(objectives[:,j])) for j,p in enumerate(POLICIES)}
    context=dict(ii=ii,jj=jj,Kcc=Kcc,Kqc=Kqc,mq=mq,mc=mc,noise=noise,vn=vn,
                 weights=weights,reference=reference,xc=xc,xq=xq)
    return allrisk,objectives,choices,context

def pair_risks(cx,qc,x,q0,prior):
    assert not set(cx)&{'qe','Cf','removal_fraction'} and not set(x)&{'qe','Cf','removal_fraction'}
    ii,jj=np.triu_indices(len(cx),1);ok=cx.condition_block.to_numpy()[ii]!=cx.condition_block.to_numpy()[jj];ii=ii[ok];jj=jj[ok]
    K=prior.kernel(cx,cx,'hybrid');C=prior.kernel(x,cx,'hybrid')
    mc=prior.mean(cx,'hybrid');mq=prior.mean(x,'hybrid')
    sc=cx.available_mass_per_g.to_numpy();s=x.available_mass_per_g.to_numpy()
    ra=prior.observation_noise(qc,sc,True);vq=prior.diag(x,'hybrid')+prior.observation_noise(q0,s,True)
    count=x.groupby('condition_block').excel_row.transform('size').to_numpy();w=1/count;w=w/w.sum()
    r=fraction(q0,s);J=np.c_[s*r*(1-r),1-r];weight=w[:,None]*J**2
    reference=weight.T@(mq*mq+vq);assert np.all(reference>0)
    risk=np.empty((len(ii),6));xc=np.log(sc);xq=np.log(s)
    for left in range(0,len(ii),2048):
        sl=slice(left,left+2048);i=ii[sl];j=jj[sl]
        a=K[i,i]+ra[i];b=K[j,j]+ra[j];c=K[i,j];u=C[:,i];v=C[:,j]
        # Unrestricted ordinary-kriging weight, including the actual estimator nugget.
        profile=(b[None,:]+1e-8-c[None,:]+u-v)/(a+b+2e-8-2*c)[None,:]
        stable=np.clip(profile,0,1)
        center=(xc[i]+xc[j])/2;dx=(xc[i]-xc[j])/2
        simple=.5+(xq[:,None]-center[None,:])*dx[None,:]/(4+2*dx*dx)[None,:]
        for h,w1 in enumerate([simple,stable,profile]):
            w2=1-w1
            bias=mq[:,None]-w1*mc[i][None,:]-w2*mc[j][None,:]
            value=bias*bias+vq[:,None]-2*(w1*u+w2*v)+w1*w1*a[None,:]+2*w1*w2*c[None,:]+w2*w2*b[None,:]
            assert value.min()>-1e-6
            risk[sl,2*h:2*h+2]=(weight.T@np.maximum(value,0)).T/reference
    objectives=np.c_[risk[:,:4].max(1),risk[:,:4].mean(1),risk[:,[0,1,4,5]].max(1)]
    return ii,jj,risk,objectives,dict(K=K,C=C,mean_c=mc,mean_q=mq,noise_a=ra,variance_q=vq,weights=weight,reference=reference)

def check_pair(cx,qc,x,q0,prior,ii,jj,risk,terms):
    checks=[]
    for h in np.unique(np.linspace(0,len(ii)-1,min(7,len(ii))).astype(int)):
        i,j=int(ii[h]),int(jj[h]);ids=[i,j]
        B=terms['K'][np.ix_(ids,ids)]+np.diag(terms['noise_a'][ids]);C=terms['C'][:,ids]
        convex=convex_weights(B+1e-8*np.eye(2),C)
        w1=np.clip((B[1,1]+1e-8-B[0,1]+C[:,0]-C[:,1])/(B[0,0]+B[1,1]+2e-8-2*B[0,1]),0,1)
        assert np.max(np.abs(convex[:,0]-w1))<1e-7
        variance=terms['variance_q']-2*np.sum(convex*C,axis=1)+np.einsum('ij,jk,ik->i',convex,B,convex)
        bias=terms['mean_q']-convex@terms['mean_c'][ids]
        direct=terms['weights'].T@(bias*bias+variance)/terms['reference']
        error=float(np.max(np.abs(direct-risk[h,2:4])));assert error<1e-7
        checks.append({'pair_index':int(h),'convex_active_set_weight_error':float(np.max(np.abs(convex[:,0]-w1))),
                       'matrix_risk_error':error})
    return checks

def choose_batch_pair(policy,candidates,qc,query,qq,prior):
    if policy in BATCH_POLICIES[:3]:
        ii,jj,risks,objectives,terms=pair_risks(candidates,qc,query,qq,prior)
        column=BATCH_POLICIES.index(policy);best=int(np.argmin(objectives[:,column]))
    elif policy in BATCH_POLICIES[3:]:
        risks,objectives,choices,terms=pair_scores(candidates,qc,query,qq,prior)
        column=BATCH_POLICIES[3:].index(policy);best=choices[POLICIES[column]]
        ii,jj=terms['ii'],terms['jj']
    else:raise ValueError('Unknown batch policy: '+policy)
    return [int(ii[best]),int(jj[best])],{'objective':float(objectives[best,column]),'legal_pairs':len(ii),'risk':risks[best].tolist()}
