# Model Analysis Report

## Model Performance Summary

- Best performing model: advanced_random_forest (R² = 0.4480)
- Average model performance: R² = 0.1784

Detailed Performance Metrics:

                                  r2       rmse        mae

baseline_arima -0.002232 14.171105 11.208084
baseline_lasso 0.042559 13.850828 11.103802
baseline_linear 0.043282 13.845593 11.108425
baseline_ridge 0.043240 13.845899 11.108107
advanced_gradient_boosting 0.444643 10.548862 6.701819
advanced_random_forest 0.447979 10.517134 6.632517
advanced_svr 0.229234 12.427420 8.718299

## Feature Importance Analysis

                     advanced_gradient_boosting  advanced_random_forest

Biomass Energy 0.000000 0.000000
Geothermal Energy 0.875387 0.879646
Hydroelectric Power 0.000000 0.000000
Solar Energy 0.124613 0.120354
Wind Energy 0.000000 0.000000