"""Matched calibration controls under the same revealed assay records."""
from .methods import calibrate
from .profile import profile_predict
from .constraints import convex_predict
from .mixtures import mixture_calibrate
METHODS=['Source_bound','Ln_ridge_bound','Logit_ridge','EB_logit_kernel','Profile_hybrid','Profile_zero_mean',
         'Convex_contrast','Convex_source_mean','SCM_evidence','SCM_contrast_evidence','SCM_contrast_moment']

def predict(method,x,q0,ax,qa,ya,priors,mix,details=False):
    if method.startswith('Profile_'):return profile_predict(method,x,q0,ax,qa,ya,priors['logit'],details)
    if method.startswith('Convex_'):return convex_predict(method,x,q0,ax,qa,ya,priors['logit'],details)
    if method.startswith('SCM_'):return mixture_calibrate(method,x,q0,ax,qa,ya,mix,details)
    p=calibrate(method,x,q0,ax,qa,ya,priors)
    return (p,{}) if details else p
