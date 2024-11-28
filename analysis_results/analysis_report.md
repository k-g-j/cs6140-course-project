# Model Analysis Report


## 1. Model Performance Summary

- **Best performing model:** advanced_random_forest (R² = 0.4480)
- **Average R² across models:** 0.1784

### Detailed Performance Metrics:

|                            |          r2 |    rmse |      mae |
|:---------------------------|------------:|--------:|---------:|
| baseline_arima             | -0.00223177 | 14.1711 | 11.2081  |
| baseline_lasso             |  0.0425587  | 13.8508 | 11.1038  |
| baseline_linear            |  0.0432823  | 13.8456 | 11.1084  |
| baseline_ridge             |  0.04324    | 13.8459 | 11.1081  |
| advanced_gradient_boosting |  0.444643   | 10.5489 |  6.70182 |
| advanced_random_forest     |  0.447979   | 10.5171 |  6.63252 |
| advanced_svr               |  0.229234   | 12.4274 |  8.7183  |

## 2. Feature Importance Analysis

|                     |   advanced_gradient_boosting |   advanced_random_forest |
|:--------------------|-----------------------------:|-------------------------:|
| Biomass Energy      |                     0        |                 0        |
| Geothermal Energy   |                     0.875387 |                 0.879646 |
| Hydroelectric Power |                     0        |                 0        |
| Solar Energy        |                     0.124613 |                 0.120354 |
| Wind Energy         |                     0        |                 0        |

## 3. Ablation Study Insights


### Data Volume

- **Sizes:** size_20, size_40, size_60, size_80, size_100
- **R² Scores:** 0.4058, 0.3363, 0.2959, 0.3971, 0.4446


### Feature Importance

- **Overall Impact:** 0.00%
- **Average Impact:** 0.00%
- **Most Impacted Feature Group:** economic


### Model Complexity

- **Best Configuration:** rf_n100_d10
- **Best R² Score:** 0.4480
