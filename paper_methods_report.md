# Methods and Results Report for the Wildfire Prediction Project

## Purpose and Study Design

This project develops and evaluates machine-learning models for predicting wildfire-related satellite fire activity at the grid-cell-day scale. The prediction target is whether a spatial grid cell is fire-positive on a given day according to filtered MODIS/FIRMS active-fire detections. The output should therefore be interpreted as the probability of observing a fire-positive grid-cell-day, not as a direct count of independent ignition events, burned area, fire duration, or fire severity.

The study is designed around a retrospective supervised-learning framework. Historical satellite fire detections define positive examples. Negative examples are sampled from the same geographic domain, with several strata chosen to make the classification task realistic and informative. Candidate predictors combine meteorological history, fire-weather indices, topography, land-cover information, ecological context, anthropogenic proximity indicators, and recent fire history. Models are trained on earlier years, tuned on later validation years, and tested on the most recent held-out years.

Two evaluation regimes are used and should be kept conceptually separate:

- A sampled case-control evaluation, where negatives are deliberately undersampled. This setting is useful for comparing models and studying feature contributions, but its event prevalence is much higher than deployment prevalence.
- A deployment-style full-grid evaluation, where model predictions are assessed on a dense country grid over time with sampling weights. This setting better represents real operational rarity and calibration, but it is currently available for the Russian Federation evaluation grid rather than the full multi-country training domain.

The main scientific question is whether the combined use of dynamic weather history, static landscape context, and learned spatiotemporal representations can improve detection of fire-prone cell-days relative to simpler weather-only, fire-weather-only, linear, and tree-based baselines.

## Observation Unit, Domain, and Temporal Splits

The basic observation unit is a daily grid cell. Fire detections and all predictors are mapped to a regular grid with approximately 0.1 deg spacing in latitude and longitude, corresponding to roughly 11 km in the north-south direction. Each row represents one grid cell on one calendar day.

The broad sampled dataset covers northern Eurasian and neighboring regions, with grid coordinates approximately spanning 35 to 75 deg latitude and 6 to 179 deg longitude. The sampled training data include many countries across Russia, Europe, Central Asia, East Asia, and adjacent regions. The current full-grid calibrated evaluation is restricted to the Russian Federation because it requires much denser deployment-grid feature generation.

The principal retrospective split is chronological:

- Training: 2001-2018.
- Validation: 2019-2020.
- Test: 2021-2025.

This chronological split is important because fire activity, weather regimes, land use, observation conditions, and reporting artifacts can vary substantially over time. A random split would overstate generalization by mixing nearby dates and related fire episodes across training and test sets.

For the full-grid calibrated evaluation, a separate calibration period and test period are used:

- Calibration grid: 2021.
- Full-grid test: 2022 through 2 October 2025.

The calibration year is used only to adjust probability scale and expected event counts before evaluating the held-out full-grid test period.

## Fire Data and Target Construction

### Satellite Fire Detections

Fire labels are derived from MODIS/FIRMS active-fire detections. These detections provide the empirical basis for identifying fire-positive cell-days. Detections are filtered before label construction to reduce false positives and persistent non-wildfire sources.

The standard detection filter uses:

- Brightness threshold: at least 380.
- Confidence threshold: at least 0.85.

Because high-latitude fires can have different observation conditions and lower apparent brightness, relaxed thresholds are used in high-latitude regions:

- High-latitude brightness threshold: at least 360.
- High-latitude confidence threshold: at least 0.70.

An even more permissive adjustment is applied in part of the far-northern western domain, where the filtering is relaxed further to reduce omission of valid high-latitude fire detections. Confidence values are normalized internally so that the same logic works whether confidence is expressed as a fraction or percentage.

Persistent stationary detections are removed using a stationary-point catalogue. This step is intended to reduce contamination from recurring industrial heat sources, gas flares, or other stable non-wildfire thermal anomalies.

### Gridding and Daily Aggregation

After filtering, fire detections are rounded to the 0.1 deg grid. Detections are grouped by rounded latitude, rounded longitude, and date. A grid-cell-day is considered fire-positive if at least one retained detection falls within that grouped cell-day after expansion.

Dense or multiple detections in a cell are handled with a small spatial expansion. If more than one retained detection occurs within the same cell-day group, the central cell is retained as positive and neighboring cells in the four cardinal directions are also marked positive. This creates a limited spatial footprint around dense detections. The expansion reduces overly brittle point matching and better reflects the fact that an active fire observed by satellite may influence nearby cells at the 0.1 deg resolution.

The resulting target is binary:

- Positive: at least one retained and expanded fire signal in the grid cell on that date.
- Negative: no retained fire signal in the grid cell on that date, subject to negative-sampling rules.

The target is best interpreted as "satellite-observed active-fire presence at the grid-cell-day scale." It should not be interpreted as a unique ignition count.

## Negative Sampling and Label Uncertainty

### Motivation

True deployment data are extremely imbalanced: almost all grid-cell-days have no detected fire. Training directly on all non-fire grid-cell-days would be computationally expensive and would yield a dataset dominated by very easy negatives. Therefore, the project uses stratified negative sampling for the main training and sampled evaluation matrix.

The sampled dataset targets a positive fraction of approximately 0.15. This produces a much denser event rate than deployment reality, which improves learning and diagnostic comparisons but requires careful interpretation of precision, F1, and PR-AUC.

### Negative Strata

Negative examples are sampled from several strata:

- Near-fire hard negatives: non-fire cells close to recent positive cells, but outside an exclusion buffer.
- Same-season negatives: cells sampled from similar months or seasons.
- Same-ecoregion negatives: cells matched by ecological region where possible.
- Same burnable-landcover negatives: cells matched by land-cover and vegetation context.
- Random background negatives: broader geographic background examples.

The approximate sampling weights are:

- Near-fire hard negatives: 10%.
- Same-season negatives: 20%.
- Same-ecoregion negatives: 20%.
- Same burnable-landcover negatives: 20%.
- Random background negatives: 30%.

Candidate negative cells are generated within country geometries and coordinate bounds. They are enriched with ecological and land-cover fields where available so that negatives can be matched to plausible fire environments rather than being drawn only from obviously non-burnable or irrelevant areas.

To avoid ambiguous labels, negatives are excluded if they fall within one grid cell and one day of a positive example. Near-fire hard negatives are sampled farther away, typically 2 to 5 cells from positives and within a plus/minus 7 day window. This creates difficult non-fire examples near active fire conditions while avoiding direct overlap with the positive label.

### Soft Labels

The target matrix also supports soft labels for near-fire uncertainty. Hard positives remain label 1. Ordinary negatives remain label 0. Near-fire hard negatives can receive a partial positive label up to 0.35, with the value decaying exponentially with spatial and temporal distance from the nearest positive detection.

This design reflects uncertainty in satellite fire geolocation, grid discretization, spread between adjacent cells, and timing near active fire episodes. However, the principal reported classification metrics are computed against hard binary labels.

## Recent Fire-History Features

Recent historical fire activity is included as a predictor with an embargo to reduce direct leakage from the immediate target period. Counts are computed over local neighborhoods with radii of 0, 1, and 2 grid cells. The history windows include:

- A recent monthly-scale window ending more than one month before the target day.
- A yearly-scale window representing longer-term local fire tendency.

These features are intended to capture persistent spatial fire risk, such as recurrent fire-prone landscapes, human ignition patterns, land management regimes, or regional fuel and climate conditions. The immediate month before the target is not used as a direct predictor, reducing the chance that ongoing fires are trivially re-identified as future fires.

## Predictor Data Sources

### Meteorological Data

The main dynamic weather predictors are derived from ECMWF seasonal meteorological data. The project uses daily historical sequences for:

- 2 m air temperature.
- 2 m dewpoint temperature.
- Total precipitation.
- Soil temperature at level 1.

For each grid-cell-day, the system extracts a retrospective sequence of daily weather values, typically 128 days long. The extraction uses the nearest available climate grid point and aligns data to daily valid times. This weather history is then summarized into tabular statistics for tree-based and linear models, and preserved as a temporal tensor for neural models.

The project also contains experiments comparing alternative meteorological sources, including ERA5-style inputs and precipitation-removed variants. The paper-ready operational comparison is strongest for the ECMWF/SEAS5-based workflow, while complete full-schema parity for ERA5 remains more limited.

### Fire-Weather Indices

Fire-weather index variables are included as additional predictors. These include statistics and temporal summaries for indices such as:

- Drought code.
- Buildup index.
- Fire danger severity rating.
- Fine fuel moisture code.
- Fire weather index.

Monthly fire-weather fields are interpolated or sampled to the working grid and summarized using statistics such as mean, standard deviation, minimum, maximum, median, and trend terms. These variables provide physically interpretable fire-danger information and serve as a strong domain baseline.

### Topography and Terrain

Topographic predictors include elevation and local terrain summaries. For each grid point, the workflow computes point elevation and neighborhood statistics such as local minimum, maximum, mean, standard deviation, and gradient-related summaries. Neighborhoods at approximately 0.1 and 0.2 deg scales are used.

Topography can influence fire through slope, aspect-related microclimate, accessibility, fuel distribution, snow persistence, and vegetation structure. In this project, topographic features are treated as static predictors.

### Land Cover, Vegetation, Soil, and Coastline

Static land-surface predictors include:

- Forest cover and tree-cover information.
- Land-sea mask information.
- Low- and high-vegetation type.
- Soil type.
- Population raster information.
- Distance to coastline and coastline-derived inland/sea distance features.

Rows are filtered using the land-water mask so that the feature matrix follows the project's retained land/water criterion. Missing categorical soil or vegetation classes are handled with explicit missing or sentinel categories rather than dropping all affected rows.

### Ecoregions

WWF terrestrial ecoregion data are used to assign ecological context to grid cells. The main ecoregion predictors include ecoregion name and realm. These variables provide broad ecological stratification and help represent spatial variation in vegetation, climate, and fuel regimes.

### Anthropogenic Proximity Features

Anthropogenic predictors are included because human activity often influences ignition probability, suppression access, and observation context. The project uses:

- OpenStreetMap-derived drivable road presence.
- Distance to nearest road.
- Gaussian-smoothed road-density fields at approximately 5, 10, and 25 km scales.
- Night-light radiance and quality variables.
- Gaussian-smoothed night-light density fields at approximately 5, 10, and 25 km scales.
- Distance to nearest night-light source.
- Population density or population-related raster features.

Night-light predictors are derived from annual satellite night-light products, including Black Marble-style radiance features. When annual data are matched to a target date, the nearest available year is selected in a way that avoids unnecessary forward-looking information when a tie occurs.

## Feature Engineering for Tabular Models

For tabular models, the 128-day meteorological history is transformed into engineered predictors. These include:

- Lag features at 7, 14, 30, 90, and 120 days.
- Rolling-window summaries over 7, 14, 30, 90, and 120 days.
- Exponentially weighted moving averages over corresponding spans.
- Difference and percentage-change features for temperature and dewpoint variables.
- Trend features over 21-day and 90-day windows.
- Extended rolling statistics such as minimum, maximum, and median.

Different meteorological variables use different summary families. Temperature and dewpoint include rolling statistics, differences, percentage changes, trends, exponentially weighted summaries, and extended rolling summaries. Precipitation is represented through rolling, exponentially weighted, and extended rolling statistics. Soil temperature includes rolling, trend, exponentially weighted, and extended rolling summaries.

This representation gives tree-based and linear models access to both short-term fire-weather conditions and longer antecedent moisture or heat patterns.

Categorical variables such as country, ecoregion, soil type, and vegetation classes are kept as categorical predictors for models that support them. For models that require numerical encoding, categories are encoded using training-derived mappings, with safeguards for unseen categories.

Direct latitude and longitude predictors are disabled in the principal no-location tabular revision suite to reduce reliance on raw geographic memorization. However, geographic context can still enter through country, ecoregion, land-surface, terrain, coastline, and history-derived predictors. Neural spatial models include coordinate encodings in some experiments, described below.

## Neural Data Preparation

Neural models use a structured representation of the same prediction problem. The dynamic meteorological input is arranged as a sequence with shape conceptually corresponding to:

- Number of examples.
- Number of days in the historical sequence.
- Number of meteorological channels.

For spatial climate neural models, each day also includes a local 3 by 3 spatial patch around the target grid cell. This allows the model to learn small-scale spatial gradients and neighboring weather context rather than relying only on a single nearest climate point.

Dynamic variables are imputed using training-set channel medians and standardized using training-set means and standard deviations. Static numerical variables are median-imputed and standardized. Missingness indicators are appended for static variables where relevant. Categorical variables are either one-hot encoded for simpler models or represented with learned embeddings in embedding-based neural architectures.

Some neural experiments add coordinate encodings to the static feature branch, including scaled latitude and longitude and sinusoidal latitude/longitude transformations. These provide spatial position information in a smoother form than raw coordinates.

## Model Families

### CatBoost Gradient Boosting

CatBoost is the main tabular nonlinear model. It is well suited to this problem because it can handle heterogeneous numerical and categorical predictors, nonlinear interactions, missing values, and high-dimensional engineered weather features. It is trained as a binary classifier with validation monitoring and early stopping.

CatBoost is evaluated in several feature configurations:

- Full feature set.
- Weather-only feature set.
- Fire-weather-index-only feature set.
- Static-only feature set.
- Dynamic weather plus fire-weather-index feature set.
- Full feature set with anthropogenic features removed.
- Full feature set with ecoregion variables removed.
- Full feature set with geography/location variables removed.
- Full feature set with fuel, ecoregion, and vegetation variables removed.
- Full feature set with terrain variables removed.
- Full feature set with longer weather-history windows removed.
- Full feature set with Gaussian-smoothed anthropogenic rasters removed.
- Full feature set with seasonality or recent-fire-history groups removed where available.

These ablations are used to test which classes of predictors contribute most to discrimination.

### Random Forest

The random forest baseline is a nonlinear tree ensemble trained on the tabular feature matrix. It provides a comparison to CatBoost using a more classical bagged-tree approach. Because random forests are computationally heavier on very large feature matrices, training is capped with stratified sampling for feasibility. Class weighting is used to reduce the impact of class imbalance.

### Logistic Regression with Stochastic Optimization

The linear logistic baseline uses stochastic gradient descent with log-loss. Numerical predictors are imputed and standardized; categorical predictors are encoded using training-derived ordinal or one-hot-style representations depending on the experiment. This model provides a low-complexity reference point and helps quantify how much nonlinear modeling improves performance.

### Poisson Point-Process Generalized Linear Model

The Poisson point-process baseline models expected fire counts or intensity rather than directly modeling only binary probability. Predicted intensity is converted to a probability using the relationship:

`P(event) = 1 - exp(-lambda)`

where lambda is the predicted event intensity for a grid-cell-day under equal exposure. This baseline is useful because the underlying scientific problem resembles rare-event intensity estimation over space and time.

### Fire-Weather and Weather-Only Baselines

Two restricted CatBoost baselines are especially important:

- Weather-only CatBoost, using meteorological and related temporal-history predictors.
- Fire-weather-index-only CatBoost, using fire-weather variables only.

These baselines assess how much predictive power is available from physical fire-weather information alone, and how much is added by static geography, ecology, land cover, anthropogenic context, and richer feature fusion.

## Neural Model Architectures

Several neural architectures are evaluated on the sampled case-control neural dataset.

### Minimal MLP

The minimal multilayer perceptron provides a compact baseline using flattened or summarized dynamic inputs together with static and categorical inputs. It tests whether a simple feed-forward network can learn useful nonlinear interactions from the prepared feature tensors.

### FT-Transformer

The FT-Transformer represents dynamic summaries, static numerical inputs, and categorical inputs as tokens processed by a transformer encoder. A classification token is used to aggregate information before the final prediction head. This architecture is designed for heterogeneous tabular data and can model interactions among feature groups.

### LSTM-Based Models

Several recurrent architectures process the meteorological sequence:

- LSTM with static concatenation: the sequence is encoded by an LSTM and fused with a static branch.
- LSTM with attention: the model learns attention weights over daily sequence outputs before fusion.
- LSTM gated mixture of experts: a bidirectional LSTM and static branch feed a learned expert mixture, allowing the model to combine multiple specialized decision functions.

These models test whether recurrent sequence learning can capture temporal patterns in weather history that are not fully represented by hand-crafted rolling features.

### Temporal Convolutional Sequence Network

The temporal sequence network uses dilated temporal convolution blocks. Each block combines multiple kernel sizes and dilation rates, allowing the model to capture both short-term and longer-term weather dependencies. Residual connections, normalization, nonlinear activation, dropout, and attention pooling are used to stabilize learning and summarize the sequence.

### Spatial Climate Temporal Sequence Network

The strongest neural architecture is the spatial climate temporal sequence network with static and categorical fusion. For each day in the 128-day history, the model first encodes a local 3 by 3 climate patch using spatial convolution. The daily spatial embeddings are then passed through a dilated temporal convolutional sequence network. Attention pooling summarizes the historical sequence, and the result is fused with static numerical features and categorical embeddings.

This architecture directly represents the fact that fire risk is not only a function of the exact grid-cell weather value, but also of nearby spatial climate structure, gradients, and regional context.

## Neural Training Procedure

Neural models are trained with mini-batch optimization. The main spatial climate temporal sequence model uses:

- A 128-day meteorological history.
- Four dynamic channels in the full version: temperature, dewpoint, precipitation, and soil temperature.
- A 3 by 3 local spatial patch for each day.
- Learned embeddings for categorical variables.
- Static numerical fusion.
- Focal loss to emphasize difficult and minority positive examples.
- AdamW-style optimization with weight decay.
- Cosine learning-rate scheduling.
- Gradient clipping.
- Early stopping based on validation performance.

Validation metrics include average precision and F1. The final threshold used for precision, recall, and F1 is selected on the validation set and then applied unchanged to the test set.

Full-grid calibrated neural evaluation is not yet available for all neural architectures because it requires a deployment-grid tensor-generation adapter. Therefore, neural results are reported as sampled case-control results, not deployment-calibrated probabilities.

## Thresholding, Metrics, and Uncertainty

For sampled case-control evaluation, models output continuous scores or probabilities. A classification threshold is chosen using the validation set by maximizing F1. The same threshold is then applied to the test set and to regional or annual subsets. This prevents test-set threshold tuning.

The primary sampled metrics are:

- Precision.
- Recall.
- F1.
- Average precision, also reported as PR-AUC.
- ROC-AUC.
- Brier score.

Uncertainty estimates for key sampled metrics are obtained using stratified bootstrap resampling of saved predictions. These errors should be interpreted as empirical sampling variability within the evaluated prediction table, not as full uncertainty over all possible data sources and modeling choices.

Regional sampled evaluations are reported for:

- Eastern Siberia.
- Far East.
- Central Asia.
- Europe.

These regions differ substantially in fire prevalence, ecology, climate, land use, and sample size.

## Full-Grid Calibrated Evaluation

The full-grid evaluation is designed to approximate operational deployment more closely. A dense 0.1 deg country grid is constructed within country geometry, crossed with all dates in the evaluation period, and labeled using the same MODIS/FIRMS target rules. Because a fully exhaustive grid is very large, the current evaluation uses weighted grid sampling:

- All positive grid-cell-days are retained.
- A small fraction of negative grid-cell-days is sampled.
- Sampled negatives receive inverse-probability weights.
- Sampling is stratified by country and month.

This weighted design estimates metrics over the deployment grid while keeping computation feasible.

Raw sampled-model probabilities are poorly calibrated for deployment because the training and sampled test prevalence are far higher than true grid prevalence. Therefore, probabilities are recalibrated using a prior-offset monthly calibration method. This method adjusts score intercepts so that expected weighted event counts align better with observed weighted counts during the calibration period. The calibrated model is then evaluated on the held-out full-grid test period.

Full-grid evaluation reports:

- Weighted average precision.
- Weighted ROC-AUC.
- Weighted Brier score.
- Weighted log-loss.
- Calibration slope and intercept.
- Expected-to-observed event-count ratio.
- Daily count mean absolute error.
- Reliability-bin summaries.
- Risk concentration in the top-scored grid fractions.
- Spatial tolerance results at exact, neighborhood, and coarser-grid scales.

Because true event prevalence is extremely low, full-grid PR-AUC and maximum F1 values are numerically much smaller than sampled case-control values. This is expected and should not be interpreted as a model failure by itself. Risk lift over the base rate is more informative in the rare-event deployment setting.

## Dataset Summary

The sampled case-control dataset contains:

| Split or region | Years | Cell-days | Positives | Negatives | Positive rate |
|---|---:|---:|---:|---:|---:|
| Global train | 2001-2018 | 1,217,788 | 176,199 | 1,041,589 | 0.1447 |
| Global validation | 2019-2020 | 150,671 | 29,123 | 121,548 | 0.1933 |
| Global test | 2021-2025 | 328,317 | 50,352 | 277,965 | 0.1534 |
| Eastern Siberia test | 2021-2025 | 141,119 | 31,318 | 109,801 | 0.2219 |
| Far East test | 2021-2025 | 11,451 | 1,232 | 10,219 | 0.1076 |
| Central Asia test | 2021-2025 | 40,288 | 2,518 | 37,770 | 0.0625 |
| Europe test | 2021-2025 | 38,999 | 3,392 | 35,607 | 0.0870 |

The sampled test set varies strongly by year:

| Year | Cell-days | Positives | Positive rate |
|---:|---:|---:|---:|
| 2021 | 88,635 | 25,006 | 0.2821 |
| 2022 | 59,796 | 5,057 | 0.0846 |
| 2023 | 62,298 | 7,077 | 0.1136 |
| 2024 | 66,372 | 9,280 | 0.1398 |
| 2025 | 51,216 | 3,932 | 0.0768 |

The full-grid deployment-style test has much lower prevalence. The weighted deployment denominator is approximately 385.9 million grid-cell-days, with 19,021 positive events and an event rate of approximately 4.93e-05. This is more than three thousand times rarer than the sampled case-control test prevalence of 0.1534.

## Sampled Case-Control Results

On the global sampled test set, the main tabular models achieve:

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.3677 | 0.7929 | 0.5024 | 0.5059 | 0.8495 | 0.1675 |
| Random Forest | 0.3499 | 0.8361 | 0.4933 | 0.4982 | 0.8579 | 0.1068 |
| Poisson point-process GLM | 0.3161 | 0.8092 | 0.4546 | 0.4519 | 0.8251 | 0.1263 |
| Weather-only CatBoost | 0.3490 | 0.7168 | 0.4695 | 0.4081 | 0.8094 | 0.2242 |
| Fire-weather-index-only CatBoost | 0.2070 | 0.9267 | 0.3384 | 0.2894 | 0.7236 | 0.2313 |
| Logistic regression | 0.2074 | 0.9757 | 0.3421 | 0.2068 | 0.6516 | 0.5842 |

CatBoost has the best global sampled F1 and PR-AUC among the principal tabular models. Random forest is very close in F1 and slightly higher in ROC-AUC, but CatBoost has higher PR-AUC and better precision at the selected validation threshold. The Poisson point-process GLM performs substantially better than the linear logistic model, indicating that intensity-based modeling is useful, but still trails the stronger nonlinear tree ensembles.

The fire-weather-index-only model reaches high recall but low precision. This suggests that fire-weather indices identify many broadly dangerous conditions, but cannot by themselves localize fire-positive grid-cell-days with enough specificity. Weather-only CatBoost performs better than fire-weather-only CatBoost, indicating that the richer meteorological history contains additional predictive signal beyond summarized fire-weather indices.

## Regional Results

Regional results show that performance is not spatially uniform.

In Eastern Siberia, all strong models perform best, reflecting both higher event prevalence and substantial fire activity in the test set. CatBoost reaches F1 = 0.5788 and PR-AUC = 0.5937, while random forest reaches F1 = 0.5803 and PR-AUC = 0.5725.

In the Far East, random forest performs best among tabular models by F1 and PR-AUC, with F1 = 0.4145 and PR-AUC = 0.3786. CatBoost remains competitive, with F1 = 0.3965 and PR-AUC = 0.3195.

In Central Asia, performance is lower because the sampled positive rate is much lower and the fire regime may differ from the dominant training signal. Random forest is strongest in this region, with F1 = 0.2227 and PR-AUC = 0.2901. CatBoost reaches F1 = 0.1959 and PR-AUC = 0.1474.

In Europe, random forest again slightly outperforms CatBoost in PR-AUC and F1, reaching F1 = 0.3072 and PR-AUC = 0.2750, compared with CatBoost F1 = 0.3031 and PR-AUC = 0.2204.

These regional differences indicate that one global threshold and one global model do not fit all fire regimes equally. Region-specific calibration or thresholds may improve applied use, but they would need to be developed without leaking test information.

## Feature-Ablation Results

The ablation results show several important patterns.

First, dynamic meteorological history is highly informative. Removing lagged weather-history features from the full CatBoost model reduces global PR-AUC from 0.5059 to 0.3623 and F1 from 0.5024 to 0.4548. This is one of the clearest signals that antecedent weather is central to prediction.

Second, fire-weather indices alone are insufficient. The fire-weather-index-only model has PR-AUC = 0.2894 and F1 = 0.3384, far below the full CatBoost model. Fire-weather indices are useful and physically interpretable, but they do not capture enough spatial, ecological, or anthropogenic variation on their own.

Third, static features alone are surprisingly informative. The static-only CatBoost model reaches PR-AUC = 0.4492 and F1 = 0.4341. This suggests that long-term spatial context, including land cover, terrain, ecology, population, and accessibility, captures a large part of baseline fire susceptibility.

Fourth, dynamic weather and fire-weather variables together produce high recall but less balanced precision. The dynamic weather plus fire-weather model reaches PR-AUC = 0.4710 and F1 = 0.3869. This indicates strong event ranking ability but less effective thresholded discrimination without static context.

Fifth, restricting climate windows to 30 days causes only a small global drop relative to the full model, with PR-AUC = 0.4952 and F1 = 0.5010. This implies that much of the immediately useful meteorological signal is contained in the recent month, although longer windows still contribute to ranking and may matter more regionally.

Sixth, removing some static groups unexpectedly improves global sampled performance in several ablations. For example, removing terrain gives PR-AUC = 0.5430 and F1 = 0.5128, higher than the nominal full CatBoost model. Removing fuel/ecoregion/vegetation or direct geography also improves global PR-AUC in the sampled test. These results should be interpreted cautiously. They may indicate redundancy, overfitting, regional confounding, or differences between sampled and deployment distributions. They do not prove that those feature groups are scientifically unimportant; rather, they show that the current full feature set can benefit from feature selection and stronger regularization.

Grouped importance analyses support the same conclusion: weather and meteorological-history variables are the strongest predictor group, with 30-day climate-window information especially important. Native model importances emphasize recent and historical fire tendency, fire-weather-index summaries, precipitation minima, and temperature maxima. These attributions are predictive, not causal.

## Neural Results

Neural models are evaluated on the sampled case-control dataset. The strongest neural architecture is the spatial climate temporal sequence network, which uses 3 by 3 climate patches, a 128-day temporal sequence, static features, and categorical embeddings.

Global sampled neural results are:

| Neural model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Spatial climate temporal sequence network | 0.5573 | 0.7982 | 0.6563 | 0.6805 |
| Spatial climate temporal sequence network without precipitation | 0.5199 | 0.7634 | 0.6186 | 0.6461 |
| Minimal MLP | 0.3616 | 0.7742 | 0.4930 | 0.4582 |
| LSTM with attention | 0.3400 | 0.7817 | 0.4739 | 0.4810 |
| LSTM gated mixture of experts | 0.3179 | 0.8696 | 0.4656 | 0.4725 |
| LSTM with static concatenation | 0.3261 | 0.7622 | 0.4568 | 0.4244 |
| FT-Transformer | 0.2888 | 0.8599 | 0.4324 | 0.4215 |
| Temporal sequence network without spatial patches | 0.2355 | 0.9166 | 0.3747 | 0.4394 |

The spatial climate temporal sequence network substantially outperforms the tabular CatBoost model on sampled PR-AUC and F1. Its PR-AUC of 0.6805 is much higher than the full CatBoost sampled PR-AUC of 0.5059, and its F1 of 0.6563 is much higher than the full CatBoost sampled F1 of 0.5024.

The no-precipitation spatial model remains strong, with PR-AUC = 0.6461 and F1 = 0.6186, but it performs below the full spatial model. This indicates that precipitation contributes meaningful information, although temperature, dewpoint, soil temperature, static context, and spatial sequence learning still carry substantial predictive signal.

Neural feature ablations show that removing the dynamic weather sequence is most damaging, reducing PR-AUC by approximately 0.242 relative to the full temporal sequence network. Permutation-based neural importance also identifies temperature spatial sequence information as especially influential. Static and categorical information remain useful, but the learned spatiotemporal weather representation is the main neural advantage.

These neural results are promising, but they should be presented as sampled case-control results until full-grid neural deployment evaluation is implemented. Because neural full-grid calibration is not yet available, the neural probabilities should not be treated as calibrated operational probabilities.

## Full-Grid Deployment-Style Results

The full-grid evaluation shows how different the rare-event deployment setting is from the sampled setting.

The full-grid weighted event rate is approximately 4.93e-05, compared with 0.1534 in the sampled test set. Consequently, sampled F1 and PR-AUC values are not deployment metrics. For CatBoost, sampled PR-AUC is 0.5059, but full-grid weighted AP is 0.001393. Although the absolute AP is small, it is about 28.3 times the base event rate, showing meaningful risk concentration.

Full-grid global metrics are:

| Model | Full-grid AP | ROC-AUC | Expected/observed count ratio |
|---|---:|---:|---:|
| CatBoost | 0.001393 | 0.8778 | 3.2405 |
| Random Forest | 0.001312 | 0.9192 | 2.7305 |
| Poisson point-process GLM | 0.001035 | 0.8724 | 2.9646 |
| Weather-only CatBoost | 0.000550 | 0.8355 | 3.5541 |
| Fire-weather-index-only CatBoost | 0.000223 | 0.8182 | 3.6483 |
| Logistic regression | 0.000152 | 0.8232 | 6178.1414 |

CatBoost has the highest full-grid AP, while random forest has the highest full-grid ROC-AUC and best daily count mean absolute error among the main tabular models. The logistic regression model is poorly calibrated in the full-grid setting, with unrealistically high mean predicted probability and extremely poor expected-to-observed count ratio.

Prior-offset calibration dramatically reduces overprediction for the nonlinear models. For CatBoost, the raw expected-to-observed count ratio is approximately 4936.7 before calibration and 3.24 after calibration. Random forest improves from approximately 1663.2 to 2.73, and the Poisson model improves from approximately 2723.9 to 2.96. Calibration therefore removes several orders of magnitude of count bias, although some overprediction remains.

Risk concentration is operationally meaningful. For CatBoost:

- The top 0.1% of grid-cell-days captures 7.35% of observed events, a lift of approximately 73.5 over random selection.
- The top 0.5% captures 17.4% of events, a lift of approximately 34.8.
- The top 1% captures 24.9% of events, a lift of approximately 24.9.
- The top 5% captures 48.8% of events, a lift of approximately 9.75.

Random forest shows similar risk concentration and slightly higher recall at the top 5%. These results suggest that even when calibrated probabilities are imperfect, model ranking can concentrate fire-positive grid-cell-days into a small fraction of the deployment grid.

Spatial tolerance also matters. For CatBoost, exact 0.1 deg cell-day AP is 0.0014 and maximum F1 is 0.0184. When evaluation is relaxed to a 3 by 3 neighborhood, AP increases to 0.0078 and maximum F1 to 0.0348. At 1 deg aggregation, AP increases to 0.0170 and maximum F1 to 0.0581. This indicates that models often identify broader high-risk areas even when exact 0.1 deg localization is difficult.

## Results Discussion

The strongest overall message is that wildfire grid-cell-day prediction benefits from combining dynamic weather history with static spatial context. Weather alone is powerful, but the best tabular results require additional information about land surface, ecology, anthropogenic access, and historical fire tendency. Fire-weather indices provide a physically meaningful baseline, but they are not sufficient for high-resolution localization.

The neural spatial climate model gives the best sampled discrimination by a large margin. Its improvement over non-spatial temporal sequence models suggests that local climate neighborhoods contain information that is lost when weather is represented only at the target cell. Fire risk can depend on nearby gradients, mesoscale climate structure, and spatial continuity of hot, dry, or wet conditions. The learned 3 by 3 spatial weather representation appears to capture such structure effectively.

The no-precipitation neural result is also informative. Removing precipitation reduces performance but does not collapse the model. This indicates that temperature, dewpoint, soil temperature, static features, categorical context, and learned temporal structure still provide strong fire-risk signal. However, precipitation remains valuable, especially as a proxy for fuel moisture and recent wetting/drying.

The sampled and full-grid evaluations tell different but complementary stories. Sampled evaluation is useful for comparing model families under a controlled class balance, while full-grid evaluation is essential for understanding operational rarity, calibration, and risk concentration. A model with strong sampled F1 can still have very small deployment F1 because the real event rate is extremely low. Therefore, deployment reporting should emphasize calibrated probabilities, expected counts, risk lift, and recall in the highest-risk fractions rather than only thresholded F1.

The full-grid results also show that calibration is not optional. Models trained on a case-control sample produce scores that are not naturally calibrated to deployment prevalence. Prior-offset calibration reduces count overprediction by orders of magnitude for CatBoost, random forest, weather-only CatBoost, fire-weather-only CatBoost, and the Poisson model. Even after calibration, expected counts remain high relative to observed counts, suggesting that further calibration, regional calibration, or spatiotemporal hierarchical calibration may be needed.

Regional variation is substantial. Eastern Siberia is predicted best, likely because the test set contains many positives and the fire regime is strongly represented. Central Asia and Europe are more difficult, potentially because of lower prevalence, different fire regimes, stronger anthropogenic heterogeneity, different land-use patterns, or fewer representative positives. This supports reporting regional metrics in the paper rather than relying only on global averages.

Some ablations in which static feature groups are removed outperform the nominal full CatBoost model. This should be discussed as evidence of redundancy and possible overfitting, not as evidence that terrain, ecology, or geography are irrelevant. Static features can encode persistent regional differences that help sampled discrimination but may also interact with temporal splits, negative-sampling design, and domain shift. A paper should describe these results carefully and distinguish predictive utility from causal interpretation.

## Limitations

The target is based on satellite active-fire detections, not ground-truth ignition records. MODIS/FIRMS detections can miss fires because of clouds, overpass timing, fire size, canopy cover, smoke, sensor limitations, or filtering thresholds. They can also include non-wildfire thermal sources if filtering is imperfect.

The positive-label expansion around dense detections improves spatial tolerance but means that predicted counts correspond to expanded fire-positive cell-days, not unique fires. Summing probabilities estimates expected positive grid-cell-days under the target definition, not burned area or ignition count.

The sampled case-control dataset has an artificial positive rate near 15%, while deployment prevalence is around 0.005% in the full-grid evaluation. Precision, F1, and PR-AUC from the sampled dataset should not be presented as operational precision, operational F1, or operational probability quality.

The full-grid calibrated evaluation currently covers the Russian Federation grid only. Results may not generalize to all countries represented in the broader sampled dataset, especially where fire regimes, land-cover classes, observation quality, or human activity patterns differ.

The strongest neural results are sampled case-control results. Full-grid calibrated neural evaluation still requires a deployment-grid tensor adapter, so neural probabilities should not yet be reported as operationally calibrated deployment probabilities.

Meteorological source comparisons are partly limited by feature-schema parity. ECMWF/SEAS5-style inputs are the most complete operational path in the current workflow, while ERA5 comparisons are present but not equally complete across all feature-generation and full-grid settings.

Feature-importance and ablation analyses are predictive diagnostics, not causal evidence. Weather, roads, night lights, ecoregions, and history features may correlate with fire occurrence for many reasons, including observation bias, accessibility, suppression activity, land management, fuel distribution, and regional climate.

## Conclusions

This project builds a comprehensive wildfire grid-cell-day prediction system using satellite fire detections, dynamic meteorological histories, fire-weather indices, static land-surface variables, ecological context, anthropogenic proximity indicators, and recent fire history.

The best tabular models are nonlinear tree ensembles, especially CatBoost and random forest. On the sampled global test set, CatBoost reaches F1 = 0.5024 and PR-AUC = 0.5059, outperforming weather-only, fire-weather-only, Poisson, and logistic baselines. Random forest is close in sampled performance and has strong full-grid ROC-AUC and count behavior.

The strongest sampled model overall is the spatial climate temporal sequence neural network. By learning from 128-day sequences of local 3 by 3 climate patches and fusing them with static and categorical context, it reaches F1 = 0.6563 and PR-AUC = 0.6805 on the sampled global test set. This is the clearest evidence that learned spatiotemporal weather representations add substantial predictive value.

Full-grid evaluation shows that deployment is an extreme rare-event problem. Absolute AP and F1 are low because the event rate is approximately 4.93e-05, but the best models provide meaningful risk concentration. CatBoost achieves the highest full-grid AP and captures approximately 7.35% of events in the top 0.1% of grid-cell-days, corresponding to a lift of about 73.5 over random selection.

Calibration is essential. Case-control-trained models greatly overpredict deployment counts before calibration, and monthly prior-offset calibration reduces this bias by orders of magnitude. Further work should focus on stronger deployment calibration, full-grid neural evaluation, region-specific calibration, and clearer separation between fire-positive cell-day prediction and independent ignition or burned-area modeling.

For a paper, the most defensible central claim is that multi-source spatiotemporal modeling substantially improves retrospective prediction of satellite-observed fire-positive grid-cell-days, with dynamic weather history as the dominant signal, static landscape context as an important modifier, and spatial neural weather encoders offering the strongest sampled discrimination. Operational use should emphasize calibrated risk ranking and high-risk-area prioritization rather than interpreting raw sampled probabilities as deployment probabilities.
