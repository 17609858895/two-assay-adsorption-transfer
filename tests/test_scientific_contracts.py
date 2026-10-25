"""Tests of the release's mathematical and information contracts."""
from pathlib import Path
import sys,unittest
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
from synthetic_demo import synthetic_data,example_context,run
from adsorption_transfer.constraints import convex_weights
from adsorption_transfer.methods import BudgetedLabels,fraction,inverse_prediction,latent_observations
from adsorption_transfer.protocol import prepare_protocol,check_protocol
from adsorption_transfer.design import pair_risks,check_pair
from adsorption_transfer.metrics import metric_sums,derive,paired_material_bootstrap
from adsorption_transfer.influence import operators

class ScientificContracts(unittest.TestCase):
    def test_active_set_matches_independent_numerical_optimizer(self):
        rng=np.random.default_rng(219)
        A=rng.normal(size=(3,3));B=A@A.T+.2*np.eye(3);C=rng.normal(size=(9,3))
        W=convex_weights(B,C)
        for row,c in zip(W,C):
            result=minimize(lambda w:w@B@w-2*w@c,np.ones(3)/3,method='SLSQP',bounds=[(0,1)]*3,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-12,'maxiter':500})
            self.assertTrue(result.success)
            self.assertAlmostEqual(row@B@row-2*row@c,result.fun,places=8)
        np.testing.assert_allclose(W.sum(1),1,atol=1e-12)
        self.assertGreaterEqual(float(W.min()),0)

    def test_protocol_ignores_outcomes_and_row_order(self):
        d=synthetic_data();a,s,_=prepare_protocol(d,repeats=2)
        poisoned=d.sample(frac=1,random_state=555).copy();poisoned['qe']=1e12
        b,_,_=prepare_protocol(poisoned,repeats=2)
        self.assertTrue(a.equals(b));self.assertTrue(check_protocol(a,s).all_checks_pass.all())

    def test_actual_estimator_risk_matches_matrix_algebra(self):
        _,_,prior,cx,qc,qx,qq,_=example_context()
        ii,jj,risk,_,terms=pair_risks(cx,qc,qx,qq,prior)
        self.assertEqual(len(ii),90)
        checks=check_pair(cx,qc,qx,qq,prior,ii,jj,risk,terms)
        self.assertLess(max(c['matrix_risk_error'] for c in checks),1e-7)
        with self.assertRaises(AssertionError):pair_risks(cx.assign(qe=123),qc,qx,qq,prior)
        with self.assertRaises(AssertionError):pair_risks(cx,qc,qx.assign(Cf=123),qq,prior)

    def test_budget_and_distinct_blocks(self):
        oracle=BudgetedLabels([1,2,3],[1,2,1],[3.,4.,5.],max_k=2)
        self.assertEqual(oracle.reveal(0),3.)
        with self.assertRaises(AssertionError):oracle.reveal(2)
        self.assertEqual(oracle.reveal(1),4.)
        with self.assertRaises(AssertionError):oracle.reveal(0)

    def test_latent_contraction_and_bounded_readout(self):
        _,_,prior,cx,qc,qx,qq,_=example_context();ax=cx.iloc[[0,3]];qa=qc[[0,3]]
        W,_=operators(qx,ax,qq,qa,prior)['Convex_contrast']
        z=np.array([-.5,.8]);perturb=np.array([.03,-.08]);amp=np.max(np.abs(perturb))
        self.assertLessEqual(float(np.abs(W@perturb).max()),amp+1e-12)
        s=qx.available_mass_per_g.to_numpy()
        a=inverse_prediction(qq,s,W@z,np.zeros(len(qx)),'logit')
        b=inverse_prediction(qq,s,W@(z+perturb),np.zeros(len(qx)),'logit')
        self.assertTrue(((a>0)&(a<=s)).all())
        self.assertLessEqual(float(np.abs(a/s-b/s).max()),amp/4+1e-12)
        self.assertLessEqual(float(np.abs(np.log(a)-np.log(b)).max()),amp+1e-12)

    def test_additive_metrics_and_paired_bootstrap(self):
        import pandas as pd
        rows=[]
        for material in ['A','B','C']:
            y=np.array([1.,3.,8.]);p=np.array([1.2,2.7,7.])
            rows.append({'target_material':material,**metric_sums(y,p,np.array([2.,5.,10.]))})
        d=pd.DataFrame(rows);value=derive(d.drop(columns='target_material').sum().to_dict())
        self.assertAlmostEqual(value['rmse'],float(np.sqrt(np.mean((y-p)**2))))
        comparison=paired_material_bootstrap(d,d,n_boot=100)
        self.assertEqual(comparison['rmse']['difference'],0)
        self.assertEqual(comparison['rmse']['interval'],[0,0])

    def test_end_to_end_synthetic_calibration(self):
        report=run();self.assertEqual(report['calibrators'],11);self.assertEqual(report['revealed_outcomes'],2)
        self.assertAlmostEqual(report['maximum_latent_l1_influence'],1)
        self.assertTrue(report['all_predictions_inside_supply'])

if __name__=='__main__':unittest.main()
