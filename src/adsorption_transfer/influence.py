"""Linear latent-discrepancy influence operators for two fixed assays."""
import numpy as np
from .constraints import convex_weights

def operators(x,ax,q0,qa,prior):
    sa=ax.available_mass_per_g.to_numpy();K=prior.kernel(ax,ax,'hybrid');C=prior.kernel(x,ax,'hybrid')
    B=K+np.diag(prior.observation_noise(qa,sa,True))+1e-8*np.eye(2)
    eb=np.linalg.solve(B,C.T).T;one=np.linalg.solve(B,np.ones(2));den=one.sum()
    prof=eb+(1-C@one)[:,None]*(one/den)[None,:]
    ca=np.log(sa);cq=np.log(x.available_mass_per_g.to_numpy());dx=(ca[0]-ca[1])/2
    l1=.5+(cq-ca.mean())*dx/(4+2*dx*dx);ridge=np.c_[l1,1-l1]
    return {'Logit_ridge':(ridge,np.zeros(len(x))),
        'EB_logit_kernel':(eb,prior.mean(x,'hybrid')-eb@prior.mean(ax,'hybrid')),
        'Profile_zero_mean':(prof,np.zeros(len(x))),
        'Convex_contrast':(convex_weights(B,C),np.zeros(len(x)))}
