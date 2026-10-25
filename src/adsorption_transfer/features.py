"""The common-input 25-numerical/3-categorical specification."""
BASE=['Feedstock','Pyrolysis_temp','Heating rate (oC)','Pyrolysis_time (min)',
      'C','H','O','N','Ash','Surface area','Pore volume','Average pore size',
      'inorganics','radius_pm','hydra_radius_pm','First_ionic_IE_KJ/mol',
      'Adsorption_time (min)','Ci','solution pH','rpm','Volume (L)',
      'loading (g)','adsorption_temp','Ion Concentration (M)','Anion_type','DOM']
FEATURES=BASE+['dose_calculated','available_mass_per_g']
CATS=['Feedstock','inorganics','Anion_type']
NUMS=[c for c in FEATURES if c not in CATS]
