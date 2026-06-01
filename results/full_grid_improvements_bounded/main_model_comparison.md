# Main Model Comparison

| Model                            | Feature set                       | Region          | support | positives | precision | recall | f1     | F1 error | PR-AUC | PR-AUC error | ROC-AUC | Brier  | threshold |
| -------------------------------- | --------------------------------- | --------------- | ------- | --------- | --------- | ------ | ------ | -------- | ------ | ------------ | ------- | ------ | --------- |
| Random Forest                    | full features                     | Central Asia    | 45961   | 2503      | 0.1497    | 0.7211 | 0.2479 | 0.0092   | 0.1763 | 0.0168       | 0.8231  | 0.0953 | 0.3724    |
| CatBoost                         | all features                      | Central Asia    | 45961   | 2503      | 0.1344    | 0.6001 | 0.2196 | 0.0073   | 0.1567 | 0.0158       | 0.7897  | 0.1462 | 0.4921    |
| Poisson Point-Process GLM        | full features                     | Central Asia    | 45961   | 2503      | 0.0938    | 0.7319 | 0.1663 | 0.0002   | 0.1166 | 0.0096       | 0.7345  | 0.1024 | 0.3037    |
| Logistic Regression (linear SGD) | full features                     | Central Asia    | 45961   | 2503      | 0.0838    | 0.9501 | 0.1541 | 0.0001   | 0.0830 | 0.0002       | 0.6797  | 0.5720 | 1.0000    |
| FWI-only CatBoost                | fire-weather variables only       | Central Asia    | 45961   | 2503      | 0.0662    | 0.9257 | 0.1236 | 0.0001   | 0.0728 | 0.0020       | 0.6264  | 0.2306 | 0.4773    |
| CatBoost deployment-weighted     | all features                      | Central Asia    | 45961   | 2503      | 0.0703    | 0.6480 | 0.1268 | 0.0001   | 0.0706 | 0.0005       | 0.6193  | 0.2168 | 0.4627    |
| Weather-only CatBoost            | meteorology/history features only | Central Asia    | 45961   | 2503      | 0.0691    | 0.8058 | 0.1272 | 0.0010   | 0.0693 | 0.0007       | 0.6047  | 0.2361 | 0.4821    |
| CatBoost                         | all features                      | Eastern Siberia | 128260  | 31267     | 0.4378    | 0.9185 | 0.5929 | 0.0015   | 0.6302 | 0.0039       | 0.8550  | 0.1861 | 0.4921    |
| Random Forest                    | full features                     | Eastern Siberia | 128260  | 31267     | 0.4451    | 0.8855 | 0.5924 | 0.0039   | 0.6183 | 0.0050       | 0.8491  | 0.1487 | 0.3724    |
| Poisson Point-Process GLM        | full features                     | Eastern Siberia | 128260  | 31267     | 0.3777    | 0.9452 | 0.5397 | 0.0027   | 0.5361 | 0.0104       | 0.8208  | 0.1575 | 0.3037    |
| Weather-only CatBoost            | meteorology/history features only | Eastern Siberia | 128260  | 31267     | 0.3107    | 0.9819 | 0.4721 | 0.0024   | 0.4435 | 0.0052       | 0.7611  | 0.2409 | 0.4821    |
| FWI-only CatBoost                | fire-weather variables only       | Eastern Siberia | 128260  | 31267     | 0.3100    | 0.9330 | 0.4654 | 0.0031   | 0.3904 | 0.0106       | 0.7203  | 0.2296 | 0.4773    |
| Logistic Regression (linear SGD) | full features                     | Eastern Siberia | 128260  | 31267     | 0.3268    | 0.9879 | 0.4912 | 0.0012   | 0.3262 | 0.0008       | 0.6665  | 0.5072 | 1.0000    |
| CatBoost deployment-weighted     | all features                      | Eastern Siberia | 128260  | 31267     | 0.2585    | 0.9295 | 0.4044 | 0.0003   | 0.2627 | 0.0019       | 0.5438  | 0.2322 | 0.4627    |
| CatBoost                         | all features                      | Europe          | 52062   | 2848      | 0.2282    | 0.3536 | 0.2774 | 0.0076   | 0.2049 | 0.0033       | 0.8420  | 0.0844 | 0.4921    |
| Random Forest                    | full features                     | Europe          | 52062   | 2848      | 0.2294    | 0.2932 | 0.2574 | 0.0037   | 0.1982 | 0.0082       | 0.8589  | 0.0579 | 0.3724    |
| Poisson Point-Process GLM        | full features                     | Europe          | 52062   | 2848      | 0.1443    | 0.5938 | 0.2322 | 0.0006   | 0.1616 | 0.0142       | 0.8125  | 0.0698 | 0.3037    |
| Weather-only CatBoost            | meteorology/history features only | Europe          | 52062   | 2848      | 0.0908    | 0.7978 | 0.1630 | 0.0001   | 0.1458 | 0.0051       | 0.7480  | 0.2280 | 0.4821    |
| Logistic Regression (linear SGD) | full features                     | Europe          | 52062   | 2848      | 0.1262    | 0.8862 | 0.2209 | 0.0014   | 0.1199 | 0.0010       | 0.7719  | 0.3488 | 1.0000    |
| FWI-only CatBoost                | fire-weather variables only       | Europe          | 52062   | 2848      | 0.0794    | 0.8332 | 0.1450 | 0.0022   | 0.1128 | 0.0052       | 0.7295  | 0.2143 | 0.4773    |
| CatBoost deployment-weighted     | all features                      | Europe          | 52062   | 2848      | 0.0568    | 0.1289 | 0.0788 | 0.0022   | 0.0738 | 0.0004       | 0.6318  | 0.2155 | 0.4627    |
| Random Forest                    | full features                     | Far East        | 10457   | 1230      | 0.3122    | 0.7073 | 0.4332 | 0.0012   | 0.3874 | 0.0031       | 0.8346  | 0.1140 | 0.3724    |
| CatBoost                         | all features                      | Far East        | 10457   | 1230      | 0.3035    | 0.6854 | 0.4207 | 0.0033   | 0.3490 | 0.0073       | 0.8196  | 0.1508 | 0.4921    |
| Poisson Point-Process GLM        | full features                     | Far East        | 10457   | 1230      | 0.2133    | 0.8472 | 0.3409 | 0.0004   | 0.3077 | 0.0089       | 0.7837  | 0.1301 | 0.3037    |
| FWI-only CatBoost                | fire-weather variables only       | Far East        | 10457   | 1230      | 0.1905    | 0.8089 | 0.3084 | 0.0059   | 0.3066 | 0.0009       | 0.7613  | 0.2171 | 0.4773    |
| Weather-only CatBoost            | meteorology/history features only | Far East        | 10457   | 1230      | 0.1370    | 0.8862 | 0.2374 | 0.0025   | 0.1882 | 0.0097       | 0.6440  | 0.2409 | 0.4821    |
| Logistic Regression (linear SGD) | full features                     | Far East        | 10457   | 1230      | 0.1643    | 0.9813 | 0.2815 | 0.0032   | 0.1637 | 0.0021       | 0.6585  | 0.6003 | 1.0000    |
| CatBoost deployment-weighted     | all features                      | Far East        | 10457   | 1230      | 0.1062    | 0.6683 | 0.1833 | 0.0058   | 0.1051 | 0.0021       | 0.4442  | 0.2226 | 0.4627    |
| CatBoost                         | all features                      | Global          | 321173  | 49514     | 0.3867    | 0.8212 | 0.5258 | 0.0040   | 0.5463 | 0.0039       | 0.8695  | 0.1487 | 0.4921    |
| Random Forest                    | full features                     | Global          | 321173  | 49514     | 0.3892    | 0.7939 | 0.5224 | 0.0040   | 0.5368 | 0.0104       | 0.8661  | 0.1133 | 0.3724    |
| Poisson Point-Process GLM        | full features                     | Global          | 321173  | 49514     | 0.2963    | 0.8748 | 0.4427 | 0.0003   | 0.4513 | 0.0064       | 0.8367  | 0.1237 | 0.3037    |
| Weather-only CatBoost            | meteorology/history features only | Global          | 321173  | 49514     | 0.2253    | 0.9457 | 0.3639 | 0.0009   | 0.3508 | 0.0034       | 0.7880  | 0.2357 | 0.4821    |
| FWI-only CatBoost                | fire-weather variables only       | Global          | 321173  | 49514     | 0.2130    | 0.9109 | 0.3453 | 0.0011   | 0.3106 | 0.0030       | 0.7288  | 0.2228 | 0.4773    |
| Logistic Regression (linear SGD) | full features                     | Global          | 321173  | 49514     | 0.2437    | 0.9686 | 0.3894 | 0.0021   | 0.2420 | 0.0016       | 0.7125  | 0.4763 | 1.0000    |
| CatBoost deployment-weighted     | all features                      | Global          | 321173  | 49514     | 0.1954    | 0.8523 | 0.3180 | 0.0029   | 0.1951 | 0.0016       | 0.6183  | 0.2248 | 0.4627    |
