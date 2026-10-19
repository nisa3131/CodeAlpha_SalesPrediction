# Sales Prediction using Python

**Prepared for:** nisa3131  
**Report date:** 19 October 2026 (Asia/Dhaka)  
**Repository:** `CodeAlpha_SalesPrediction`

This report uses the standard layout approved by the user: objectives, data, method, results, limitations, and reproducibility. All reported findings come from the supplied datasets and executed project analyses.


## 1  Objective

Predict Sales from TV, Radio, and Newspaper advertising measurements, comparing regression models and exploring how their predictions vary with advertising inputs. The available cross-sectional data support conditional prediction for new observations. Dated future forecasting, target segments, and online platforms cannot be evaluated because those fields are absent.


## 2  Data and provenance

Input: the user-supplied archive (3).zip, containing Advertising.csv. The task document links to https://www.kaggle.com/datasets/bumba5341/advertisingcsv. That link is cited as the assignment’s dataset reference; these analyses use the uploaded files rather than a new download. Original collection details and redistribution licensing were not independently verified. No dataset license is asserted.

| Attribute | Observed data / treatment |
| --- | --- |
| Observations | 200 |
| Predictors / target | TV, Radio, Newspaper / Sales |
| Data quality | No missing model measurements; no duplicate observations after excluding row index |
| Identifier | Verified sequential Unnamed: 0 excluded from predictors; retained for row traceability |
| Units | Observed dataset units; no verified currency or sales-volume scale |

No dated campaign, geography, customer segment, platform-detail, price, or macroeconomic controls are supplied. The analysis preserves observed values and does not assign currency, create missing variables, or interpret the index as time.


## 3  Method and evaluation

With seed 42, a fixed random 80/20 split creates 160 training and 40 untouched test rows. A chronological split is unavailable because there are no dates. All six predeclared candidates use the same five shuffled training-only cross-validation folds. Median imputation, scaling, and interaction/polynomial transformations are inside each fold’s pipeline.

Compare a mean baseline, linear regression, Ridge, a TV × Radio interaction model, quadratic Ridge, and random forest using fixed hyperparameters documented in make_models(). Lowest mean training CV RMSE selects the model before the selected estimator and baseline are evaluated once on the held-out rows. The fitted model remains trained on the 160-row training partition; it is not refit on the held-out data.


## 4  Results and marketing interpretation

| Candidate | CV RMSE ± SD | CV MAE | CV R² |
| --- | --- | --- | --- |
| Quadratic ridge (alpha=1) | 0.671 ± 0.375 | 0.441 | 0.977 |
| TV x Radio interaction | 0.968 ± 0.370 | 0.701 | 0.957 |
| Random forest | 1.021 ± 0.247 | 0.738 | 0.956 |
| Linear regression | 1.676 ± 0.297 | 1.237 | 0.870 |
| Ridge (alpha=1) | 1.677 ± 0.289 | 1.237 | 0.871 |
| Dummy mean | 5.150 ± 0.445 | 4.191 | -0.099 |

*Training-only model comparison. Fold SD is variation across five folds, not a confidence interval.*

**Selected model: Quadratic ridge (alpha=1).** The polynomial feature representation allows curvature and interactions, with Ridge regularization applied after scaling.

| Held-out model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Selected quadratic Ridge | 0.4848 | 0.6210 | 0.9878 |
| Mean baseline | 4.9625 | 5.6315 | -0.0048 |

*Untouched 40-row test set. MAE/RMSE use observed Sales units; lower is better. Higher R² is better.*

![Figure 1. Selected-model held-out predictions and residuals. Strong performance on this split does not establish future-period validity.](outputs/charts/03_heldout_predictions_and_residuals.png)

*Figure 1. Selected-model held-out predictions and residuals. Strong performance on this split does not establish future-period validity.*

| Training linear reference | Sales association per 1 SD input increase |
| --- | --- |
| TV | 3.764 |
| Radio | 2.792 |
| Newspaper | 0.056 |

*Standardized coefficients from the training-only linear reference model, holding the other observed inputs fixed.*

TV and Radio show stronger conditional linear associations than Newspaper in this training sample. The linear reference differs from the selected nonlinear predictor. These coefficients do not establish an intervention effect, and a small Newspaper coefficient does not establish a zero causal contribution.

![Figure 2. Training-only standardized coefficients. Coefficients summarize fitted associations and must not be read as causal return on advertising.](outputs/charts/04_linear_association_coefficients.png)

*Figure 2. Training-only standardized coefficients. Coefficients summarize fitted associations and must not be read as causal return on advertising.*

| Hypothetical input scenario | Predicted Sales | Δ from reference |
| --- | --- | --- |
| Observed training reference | 14.662 | +0.000 |
| TV −20% (range clipped) | 13.268 | -1.394 |
| TV +20% (range clipped) | 15.903 | +1.241 |
| Radio −20% (range clipped) | 13.759 | -0.903 |
| Radio +20% (range clipped) | 15.570 | +0.908 |
| Newspaper −20% (range clipped) | 14.643 | -0.019 |
| Newspaper +20% (range clipped) | 14.681 | +0.018 |

*Starting point: actual training source row 20. All values use observed dataset units.*

Each sensitivity scenario varies one channel by ±20%, clipped to its observed training range, while holding the others fixed. Marginal range checks do not guarantee joint observational support. There is no observed scenario outcome. These are model-based illustrations rather than causal uplift, guaranteed ROI, or an optimal budget allocation.

- Marketing next step: prioritize measured, randomized TV and Radio experiments, including an appropriate Newspaper comparison, to test the observed hypotheses.
- Collect dated spend, verified units, channel costs, revenue, segment labels, and outcomes. Predefine evaluation metrics and an adequate validation period.
- Use temporal validation once dated records are available. This dataset alone does not justify reallocating a real advertising budget.


## 5  Limitations and next steps

Only 200 observations and one 40-row holdout are available. Model selection can make the winning CV score optimistic; the test set provides an independent check within this sample. Unmeasured confounding, correlated channels, distribution shift, and absent time information limit marketing interpretation. No forecast for a named future period, causal budget effect, or assured business return is claimed. Verify dataset provenance, redistribution permission, and measurement definitions before production use.


## 6  Reproducibility

Validation recorded on 19 October 2026: the delivered notebook executed 5 code cells with 0 recorded errors. The analysis pipeline and its built-in data/evaluation checks completed successfully.

Install the pinned requirements and rerun from the repository root. The notebook calls the same analysis pipeline.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

outputs/metrics.json records source SHA-256, model settings, row split IDs, CV results, and held-out errors. Tables retain descriptive statistics, correlations, fold-level CV scores, training-only linear associations, and hypothetical inputs. heldout_predictions.csv records observed and predicted test values. Five PNG charts are generated; optional --data and --output-dir arguments support alternate local paths.
