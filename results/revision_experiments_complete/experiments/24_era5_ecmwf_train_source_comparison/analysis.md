# Analysis

- Highest ECMWF-test PR-AUC: ECMWF train -> ECMWF test (0.8250).
- Highest ECMWF-test F1: ERA5 + ECMWF train -> ECMWF test (0.6980).
- ERA5 train -> ECMWF test vs ECMWF baseline: Delta PR-AUC=-0.0994, Delta F1=-0.0322.
- ERA5 + ECMWF train -> ECMWF test vs ECMWF baseline: Delta PR-AUC=-0.0014, Delta F1=+0.0293.
- ERA5-trained performance on ECMWF test should be interpreted as input-source domain transfer, not an ERA5 retrospective upper bound.
- The mixed-source model tests whether duplicating targets with both climate sources improves operational robustness on ECMWF inputs.
