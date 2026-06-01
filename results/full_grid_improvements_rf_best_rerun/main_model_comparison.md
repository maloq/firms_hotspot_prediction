# Main Model Comparison

| Model         | Feature set   | Region          | support | positives | precision | recall | f1     | F1 error | PR-AUC | PR-AUC error | ROC-AUC | Brier  | threshold |
| ------------- | ------------- | --------------- | ------- | --------- | --------- | ------ | ------ | -------- | ------ | ------------ | ------- | ------ | --------- |
| Random Forest | full features | Central Asia    | 45961   | 2503      | 0.1018    | 0.9517 | 0.1840 | 0.0017   | 0.2601 | 0.0136       | 0.8402  | 0.0850 | 0.1423    |
| Random Forest | full features | Eastern Siberia | 128260  | 31267     | 0.3756    | 0.9500 | 0.5384 | 0.0006   | 0.5355 | 0.0041       | 0.8219  | 0.1445 | 0.1423    |
| Random Forest | full features | Europe          | 52062   | 2848      | 0.1514    | 0.8869 | 0.2586 | 0.0027   | 0.2835 | 0.0198       | 0.8735  | 0.0531 | 0.1423    |
| Random Forest | full features | Far East        | 10457   | 1230      | 0.2066    | 0.9390 | 0.3387 | 0.0011   | 0.4261 | 0.0019       | 0.8510  | 0.1055 | 0.1423    |
| Random Forest | full features | Global          | 321173  | 49514     | 0.2884    | 0.9235 | 0.4395 | 0.0014   | 0.4642 | 0.0106       | 0.8474  | 0.1080 | 0.1423    |
