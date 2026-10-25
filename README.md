# Two-assay adsorption transfer

Essential scientific code for **Influence-constrained calibration for two-assay
transfer across adsorption materials**.

The learning task retains a source predictor and uses two local measurements
to correct uptake predictions at other conditions of a held-out material.
Source-material cross-fitted discrepancies define the covariance; query-specific
nonnegative unit-sum weights control latent measurement influence, and a logit
readout respects available solute per sorbent mass. Exhaustive pair design
evaluates the actual simple, constrained and free-offset smoothers on uptake
and natural-log scales. Nonnegative kriging, Gaussian conditioning and compound
design are established ingredients; this implementation provides the tested
adsorption-transfer interface and matched controls.

## Scope

Included: mathematical calibration/design implementations, fixed material/block
protocol, 20 source-learner definitions, the common-input FT architecture port,
optional source fitting and transfer drivers, material-resampling metrics,
compact published aggregate results, plotting and meaningful synthetic checks.

Not distributed: third-party raw experimental rows, fitted model weights,
individual-query predictions, manuscript files, writing tools, private paths or
credentials. Acquire the source workbook separately as described in `data/README.md`.
The FT comparator is a common-input architecture port with source-only epoch
selection; it is not the original paper's winning checkpoint/configuration.

## Run the small deterministic example

Python 3.10 or newer is required. The study used NumPy 1.26.4, pandas 2.2.2,
SciPy 1.13.1 and scikit-learn 1.5.1. Install the dependencies and package:

```text
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python examples/synthetic_demo.py
python -m unittest discover -s tests -v
python scripts/plot_published_results.py --output outputs/matched_calibration
```

The synthetic example checks algebra and information timing; its errors are
not experimental validation or the manuscript's reported performance.

## Refit sources and evaluate the real transfer task

Obtain the pinned workbook, then use explicit input/output paths:

```text
python scripts/prepare_data.py --input /path/to/HMI_data.xlsx --output work/prepared.csv
python scripts/fit_sources.py --input work/prepared.csv --models Ridge --output work/sources
python scripts/run_transfer.py --input work/prepared.csv --source-folder work/sources/Ridge --output work/transfer/Ridge
```

`fit_sources.py` defaults to one fixed learner to avoid launching the complete
benchmark unintentionally. Select additional models with `--models`; the list
includes OLS, Ridge, ElasticNet, BayesianRidge, Huber, SVR, KernelRidge, KNN,
DecisionTree, RandomForest, ET, AdaBoost, GradientBoosting, HistGradientBoosting,
XGBoost, LGBM, CatBoost, MLP, ConvexStack and FT. Extra libraries are listed in
`requirements-sources.txt`; the study's FT runs used the PyTorch 2.13.0 CPU
build. Appropriate platform-specific PyTorch wheels may require the official
PyTorch installation instructions. No full source retraining is performed by
importing this package or running the synthetic checks.

Every target material is excluded from fitting and source-prior construction.
Three source-material folds generate OOF predictions. The convex stack repeats
its material cross-fit within each fit; FT epoch selection uses another grouped
source-only holdout and excludes the OOF held materials. Source fits are reused
across five fixed candidate/query splits. Repeated condition blocks and
duplicates stay together. Query covariates may inform the design; query and
unselected candidate outcomes do not. The six batch policies select both
records before either outcome is revealed. Sequential maximin/IVR prefixes
retain their first two/five choices at those budgets.

A complete input with 16 material labels produces 110 configurations per
learner (11 calibrators; two sequential designs at budgets two/five and six
two-assay batch designs). Across 20 learners this is the 2,200-configuration
comparison. The original paper-scale fitted weights and prediction files are
not bundled; optional refitting is computationally substantial and the full
training benchmark was not rerun as a release smoke check. Matching numerical
results also depends on the documented data, numerical libraries, order and seeds.

## Files and names

- `methods.py`: robust source-only discrepancy prior, simple/empirical-Bayes
  controls, bounded readout and sequential assay selection.
- `constraints.py`, `profile.py`, `mixtures.py`, `offset_mixtures.py`: constrained,
  free-offset and source-mixture calibration controls.
- `design.py`: all legal two-block pairs, actual-estimator risks and objectives.
- `protocol.py`, `metrics.py`, `io.py`: fixed support, additive metrics and
  prediction-file validation.
- `source_models.py`, `transformer.py`: fixed source models and grouped training
  components; optional dependencies are imported only when needed by their modules.
- `results/published_matched_summary.csv`: 60 aggregate rows for 20 learners,
  three calibrators on identical minimax-selected assay pairs.
- `SOURCE_MAP.json`: mathematical functions copied from the evaluated study code;
  file/module organization and import paths are adapted for this release.

Method identifiers in code are compact keys: `Convex_contrast` is
influence-constrained calibration, `Logit_ridge` is simple logit calibration,
`EB_logit_kernel` is empirical-Bayes calibration and `Profile_zero_mean` is
zero-mean free-offset calibration. The public design keys
`simple_convex_minimax`, `simple_convex_average` and `simple_profile_minimax`
describe their mathematical objectives, not development versions.

The fixed protocol retains the original `transfer-v2` hash salt solely to recover
the exact candidate/query split. That compatibility token does not name a method.

## Interpretation

Nonnegative unit-sum weights bound fixed-input latent-discrepancy amplification
by one. This is distinct from universal robustness to original-scale assay
noise. Matched assays isolate calibration gains; changing both source learner
and design is an end-to-end comparison. Material-resampling intervals keep all
five repeat splits together and treat 16 material labels as the resampling units;
they do not incorporate source-model refitting uncertainty. The benchmark
informed method development and is not a prospective independent validation.

No additional software license is supplied in this repository. Third-party
data and dependencies retain their own terms. Please cite the source dataset
and associated paper.
