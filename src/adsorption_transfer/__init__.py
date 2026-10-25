"""Scientific interfaces for limited-assay adsorption transfer."""
from .methods import EmpiricalPrior, BudgetedLabels, calibrate
from .constraints import convex_weights, convex_predict
from .design import pair_risks, choose_batch_pair
from .protocol import prepare_protocol, check_protocol
