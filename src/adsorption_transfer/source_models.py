"""Fixed source learners; each preprocessing/stack fit uses its source partition."""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet,BayesianRidge,HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from .features import FEATURES,CATS
try:from xgboost import XGBRegressor
except ImportError:XGBRegressor=None
try:from catboost import CatBoostRegressor
except ImportError:CatBoostRegressor=None
try:from lightgbm import LGBMRegressor
except ImportError:LGBMRegressor=None
SEED=3407
SOURCE_IDS=['ET','LGBM','FT','OLS','Ridge','ElasticNet','BayesianRidge','Huber','SVR','KernelRidge','KNN','DecisionTree','RandomForest','AdaBoost','GradientBoosting','HistGradientBoosting','XGBoost','CatBoost','MLP','ConvexStack']

def preprocessor():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATS),
        ('num', StandardScaler(), [c for c in FEATURES if c not in CATS])],
        sparse_threshold=0)

def _make_core_model(name, stack_base=False):
    if name == 'ConvexStack':
        return MaterialCrossfitConvexStack()
    estimators = {
        'OLS': lambda: RankStableOLS(rcond=1e-8),
        'Ridge': lambda: Ridge(alpha=10.),
        'ElasticNet': lambda: ElasticNet(alpha=.01, l1_ratio=.5, max_iter=10000),
        'BayesianRidge': lambda: BayesianRidge(max_iter=500),
        'Huber': lambda: HuberRegressor(epsilon=1.35, alpha=.01, max_iter=2000),
        'SVR': lambda: SVR(C=10., epsilon=.05, kernel='rbf', gamma='scale'),
        'KernelRidge': lambda: KernelRidge(alpha=1., kernel='rbf', gamma=.03),
        'KNN': lambda: KNeighborsRegressor(n_neighbors=7, weights='distance', p=2,
                                          n_jobs=2),
        'DecisionTree': lambda: DecisionTreeRegressor(max_depth=8,
                                    min_samples_leaf=5, random_state=SEED),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=250,
            min_samples_leaf=2, max_features=.9, n_jobs=2, random_state=SEED),
        'AdaBoost': lambda: AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5,
                                           random_state=SEED),
            n_estimators=200, learning_rate=.05, loss='linear', random_state=SEED),
        'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=350,
            learning_rate=.04, max_depth=3, min_samples_leaf=5,
            loss='squared_error', random_state=SEED),
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(
            max_iter=250 if not stack_base else 150, learning_rate=.06,
            max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=5.,
            early_stopping=False, random_state=SEED),
        'XGBoost': lambda: XGBRegressor(n_estimators=350, max_depth=4,
            learning_rate=.04, min_child_weight=5, reg_lambda=5.,
            subsample=1., colsample_bytree=.9, objective='reg:squarederror',
            tree_method='hist', n_jobs=2, random_state=SEED, verbosity=0),
        'CatBoost': lambda: CatBoostRegressor(iterations=400, depth=5,
            learning_rate=.04, l2_leaf_reg=5., loss_function='RMSE',
            thread_count=2, random_seed=SEED, verbose=False,
            allow_writing_files=False),
        'MLP': lambda: MLPRegressor(hidden_layer_sizes=(64, 32),
            activation='relu', solver='adam', alpha=.05, batch_size=64,
            learning_rate_init=.001, max_iter=500, early_stopping=False,
            n_iter_no_change=30, tol=1e-5, random_state=SEED),
        'ET_stack': lambda: ExtraTreesRegressor(n_estimators=100,
            min_samples_leaf=2, max_features=.9, n_jobs=2, random_state=SEED)
    }
    return make_pipeline(preprocessor(), estimators[name]())

def fit_with_groups(model, X, y, groups):
    if isinstance(model, MaterialCrossfitConvexStack):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)

class RankStableOLS(RegressorMixin, BaseEstimator):
    """Centered least squares with explicit source-design numerical rank.

    Full one-hot columns and material-level feature dependencies create exact
    and floating-point rank defects. The fixed relative SVD cutoff excludes
    numerical null directions; it is not selected using target predictions.
    """
    __module__ = 'adsorption_transfer.source_models'

    def __init__(self, rcond=1e-8):
        self.rcond = rcond

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0); self.y_mean_ = float(y.mean())
        self.coef_, residuals, self.rank_, self.singular_ = np.linalg.lstsq(
            X-self.x_mean_, y-self.y_mean_, rcond=self.rcond)
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_

class MaterialCrossfitConvexStack(RegressorMixin, BaseEstimator):
    """Source-only nested crossfit simplex stacking in log response space."""
    __module__ = 'adsorption_transfer.source_models'
    def __init__(self):
        self.base_names = ('Ridge', 'ET_stack', 'HistGradientBoosting')

    def fit(self, X, y, groups):
        y = np.asarray(y)
        groups = np.asarray(groups)
        self.training_materials_ = sorted(set(groups))
        Z = np.zeros((len(X), len(self.base_names)))
        self.fold_audit_ = []
        for f, (it, iv) in enumerate(GroupKFold(3).split(X, y, groups)):
            trmat, vamat = set(groups[it]), set(groups[iv])
            assert trmat.isdisjoint(vamat)
            self.fold_audit_.append({'fold': f, 'train_materials': sorted(trmat),
                                     'validation_materials': sorted(vamat)})
            for j, name in enumerate(self.base_names):
                est = make_model(name, stack_base=True)
                est.fit(X.iloc[it], y[it])
                Z[iv, j] = est.predict(X.iloc[iv])
        def objective(w):
            r = Z @ w - y
            return np.mean(r*r)
        result = minimize(objective, np.full(len(self.base_names),
                          1/len(self.base_names)), method='SLSQP',
                          bounds=[(0., 1.)]*len(self.base_names),
                          constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                          options={'ftol':1e-11, 'maxiter':500})
        if not result.success:
            raise RuntimeError('Simplex stacking optimization failed: '+result.message)
        self.weights_ = result.x
        assert abs(self.weights_.sum()-1) < 1e-7
        self.source_crossfit_ln_rmse_ = float(np.sqrt(objective(result.x)))
        self.models_ = []
        for name in self.base_names:
            est = make_model(name, stack_base=True)
            est.fit(X, y)
            self.models_.append(est)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models_]) @ self.weights_

def q_predictions(model, X):
    # Wide exp overflow guard is shared with v2; mass bounds are applied by
    # downstream calibrators and are not silently imposed on these source fits.
    return np.exp(np.clip(model.predict(X), -30, 30))

def _make_tree_model(name):
    proc=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATS),
        ('num',StandardScaler(),[c for c in FEATURES if c not in CATS])],sparse_threshold=0)
    if name=='ET':
        est=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,max_features=.9,n_jobs=2,random_state=3407)
    else:
        est=LGBMRegressor(n_estimators=500,learning_rate=.04,num_leaves=15,max_depth=7,
            min_child_samples=15,reg_lambda=5,colsample_bytree=.9,verbosity=-1,n_jobs=2,random_state=3407)
    return make_pipeline(proc,est)

def make_model(name,stack_base=False):
    if name=='FT':raise ValueError('Use transformer.fit_model for the FT architecture port')
    missing={'XGBoost':XGBRegressor,'CatBoost':CatBoostRegressor,'LGBM':LGBMRegressor}
    if name in missing and missing[name] is None:raise ImportError('Install requirements-sources.txt for '+name)
    return _make_tree_model(name) if name in ['ET','LGBM'] else _make_core_model(name,stack_base)
