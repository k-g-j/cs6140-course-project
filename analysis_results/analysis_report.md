# Model Analysis Report

## Model Performance Summary

- Best performing model: advanced_gradient_boosting (R² = 0.9997)
- Average model performance: R² = -0.3969

Detailed Performance Metrics:

                                  r2      rmse       mae

baseline_arima -7.773365 2.849952 2.690767
baseline_lasso -0.000640 0.962484 0.731109
baseline_linear 0.999284 0.025739 0.018587
baseline_ridge 0.999246 0.026423 0.019094
advanced_gradient_boosting 0.999671 0.017458 0.008455
advanced_random_forest 0.999556 0.020280 0.009602
advanced_svr 0.997621 0.046925 0.039046

## Feature Importance Analysis

                                     advanced_gradient_boosting  advanced_random_forest

biofuel_generation 0.028314 0.081326
geothermal_generation 0.098588 0.068616
hydro_generation 0.134513 0.122748
renewable_generation_lag_1 0.369661 0.286705
renewable_generation_lag_3 0.004067 0.013075
renewable_generation_rolling_mean_3 0.300429 0.187058
renewable_generation_rolling_mean_6 0.051430 0.178606
renewable_share_pct 0.000063 0.000151
solar_generation 0.012909 0.061655
total_energy_consumption 0.000017 0.000028
wind_generation 0.000008 0.000032