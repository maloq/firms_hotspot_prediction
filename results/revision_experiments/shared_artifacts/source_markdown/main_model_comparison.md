# Main Model Comparison

| Model                            | Feature set                       | Region          | support | positives | precision | recall | f1     | F1 error | PR-AUC | PR-AUC error | ROC-AUC | Brier  | threshold |
| -------------------------------- | --------------------------------- | --------------- | ------- | --------- | --------- | ------ | ------ | -------- | ------ | ------------ | ------- | ------ | --------- |
| Random Forest                    | full features                     | Central Asia    | 40288   | 2518      | 0.1285    | 0.8352 | 0.2227 | 0.0022   | 0.2901 | 0.0067       | 0.8108  | 0.0971 | 0.2843    |
| CatBoost                         | all features                      | Central Asia    | 40288   | 2518      | 0.1185    | 0.5643 | 0.1959 | 0.0027   | 0.1474 | 0.0025       | 0.7234  | 0.1919 | 0.4854    |
| Poisson Point-Process GLM        | full features                     | Central Asia    | 40288   | 2518      | 0.0909    | 0.5727 | 0.1569 | 0.0036   | 0.1131 | 0.0046       | 0.6548  | 0.1215 | 0.3357    |
| FWI-only CatBoost                | fire-weather variables only       | Central Asia    | 40288   | 2518      | 0.0731    | 0.9484 | 0.1358 | 0.0007   | 0.0799 | 0.0014       | 0.6116  | 0.2405 | 0.4810    |
| Logistic Regression (linear SGD) | full features                     | Central Asia    | 40288   | 2518      | 0.0738    | 0.9702 | 0.1372 | 0.0005   | 0.0736 | 0.0003       | 0.5797  | 0.7698 | 1.0000    |
| Weather-only CatBoost            | meteorology/history features only | Central Asia    | 40288   | 2518      | 0.0741    | 0.3499 | 0.1223 | 0.0019   | 0.0722 | 0.0011       | 0.5836  | 0.2336 | 0.4997    |
| CatBoost                         | all features                      | Eastern Siberia | 141119  | 31318     | 0.4262    | 0.9014 | 0.5788 | 0.0013   | 0.5937 | 0.0073       | 0.8544  | 0.1842 | 0.4854    |
| Random Forest                    | full features                     | Eastern Siberia | 141119  | 31318     | 0.4303    | 0.8906 | 0.5803 | 0.0016   | 0.5725 | 0.0058       | 0.8574  | 0.1278 | 0.2843    |
| Poisson Point-Process GLM        | full features                     | Eastern Siberia | 141119  | 31318     | 0.4006    | 0.9063 | 0.5556 | 0.0020   | 0.5433 | 0.0038       | 0.8356  | 0.1453 | 0.3357    |
| Weather-only CatBoost            | meteorology/history features only | Eastern Siberia | 141119  | 31318     | 0.4270    | 0.8169 | 0.5608 | 0.0022   | 0.5002 | 0.0048       | 0.8163  | 0.2284 | 0.4997    |
| FWI-only CatBoost                | fire-weather variables only       | Eastern Siberia | 141119  | 31318     | 0.2855    | 0.9508 | 0.4391 | 0.0006   | 0.3706 | 0.0019       | 0.7324  | 0.2346 | 0.4810    |
| Logistic Regression (linear SGD) | full features                     | Eastern Siberia | 141119  | 31318     | 0.2901    | 0.9912 | 0.4489 | 0.0004   | 0.2898 | 0.0004       | 0.6501  | 0.5484 | 1.0000    |
| Random Forest                    | full features                     | Europe          | 38999   | 3392      | 0.1979    | 0.6860 | 0.3072 | 0.0025   | 0.2750 | 0.0072       | 0.7865  | 0.0906 | 0.2843    |
| CatBoost                         | all features                      | Europe          | 38999   | 3392      | 0.2283    | 0.4508 | 0.3031 | 0.0037   | 0.2204 | 0.0070       | 0.7482  | 0.1545 | 0.4854    |
| Poisson Point-Process GLM        | full features                     | Europe          | 38999   | 3392      | 0.1644    | 0.5401 | 0.2520 | 0.0024   | 0.1879 | 0.0027       | 0.7083  | 0.1104 | 0.3357    |
| Weather-only CatBoost            | meteorology/history features only | Europe          | 38999   | 3392      | 0.2133    | 0.4826 | 0.2958 | 0.0045   | 0.1612 | 0.0026       | 0.7096  | 0.2218 | 0.4997    |
| FWI-only CatBoost                | fire-weather variables only       | Europe          | 38999   | 3392      | 0.1061    | 0.8443 | 0.1886 | 0.0007   | 0.1230 | 0.0016       | 0.6445  | 0.2318 | 0.4810    |
| Logistic Regression (linear SGD) | full features                     | Europe          | 38999   | 3392      | 0.1077    | 0.9552 | 0.1936 | 0.0002   | 0.1072 | 0.0001       | 0.6021  | 0.7008 | 1.0000    |
| Random Forest                    | full features                     | Far East        | 11451   | 1232      | 0.2740    | 0.8515 | 0.4145 | 0.0056   | 0.3786 | 0.0159       | 0.8580  | 0.1013 | 0.2843    |
| CatBoost                         | all features                      | Far East        | 11451   | 1232      | 0.2850    | 0.6518 | 0.3965 | 0.0055   | 0.3195 | 0.0046       | 0.8091  | 0.1609 | 0.4854    |
| Poisson Point-Process GLM        | full features                     | Far East        | 11451   | 1232      | 0.2183    | 0.7021 | 0.3330 | 0.0067   | 0.3031 | 0.0107       | 0.7882  | 0.1209 | 0.3357    |
| FWI-only CatBoost                | fire-weather variables only       | Far East        | 11451   | 1232      | 0.1604    | 0.8588 | 0.2702 | 0.0032   | 0.2817 | 0.0125       | 0.7493  | 0.2276 | 0.4810    |
| Weather-only CatBoost            | meteorology/history features only | Far East        | 11451   | 1232      | 0.1626    | 0.2265 | 0.1893 | 0.0046   | 0.1747 | 0.0063       | 0.6764  | 0.2281 | 0.4997    |
| Logistic Regression (linear SGD) | full features                     | Far East        | 11451   | 1232      | 0.1291    | 0.9789 | 0.2281 | 0.0013   | 0.1289 | 0.0008       | 0.5924  | 0.7189 | 1.0000    |
| CatBoost                         | all features                      | Global          | 328317  | 50352     | 0.3677    | 0.7929 | 0.5024 | 0.0025   | 0.5059 | 0.0040       | 0.8495  | 0.1675 | 0.4854    |
| Random Forest                    | full features                     | Global          | 328317  | 50352     | 0.3499    | 0.8361 | 0.4933 | 0.0013   | 0.4982 | 0.0036       | 0.8579  | 0.1068 | 0.2843    |
| Poisson Point-Process GLM        | full features                     | Global          | 328317  | 50352     | 0.3161    | 0.8092 | 0.4546 | 0.0022   | 0.4519 | 0.0085       | 0.8251  | 0.1263 | 0.3357    |
| Weather-only CatBoost            | meteorology/history features only | Global          | 328317  | 50352     | 0.3490    | 0.7168 | 0.4695 | 0.0024   | 0.4081 | 0.0034       | 0.8094  | 0.2242 | 0.4997    |
| FWI-only CatBoost                | fire-weather variables only       | Global          | 328317  | 50352     | 0.2070    | 0.9267 | 0.3384 | 0.0010   | 0.2894 | 0.0030       | 0.7236  | 0.2313 | 0.4810    |
| Logistic Regression (linear SGD) | full features                     | Global          | 328317  | 50352     | 0.2074    | 0.9757 | 0.3421 | 0.0009   | 0.2068 | 0.0006       | 0.6516  | 0.5842 | 1.0000    |
