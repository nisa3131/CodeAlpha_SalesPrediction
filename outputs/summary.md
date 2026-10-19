# Sales Prediction — Design Report

## Objective
Predict Sales from TV, Radio, and Newspaper advertising measurements using the supplied Advertising.csv. This is conditional out-of-sample regression; the data do not support a dated future sales forecast, customer segmentation, or online-platform analysis.

## Data and audit
The supplied file has 200 rows and five columns. The verified sequential `Unnamed: 0` column is excluded from predictors and retained only as a source-row identifier. There are 0 missing measurements and 0 duplicate observations ignoring the identifier. Features and Sales are numeric. Values use **observed dataset units**: the file provides no verified currency or sales-volume unit. The task document points to [Advertising.csv on Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv), while the actual input is the user's uploaded CSV. Upstream provenance and licensing were not independently verified.

Audit and source checksum: [metrics.json](metrics.json). Descriptive statistics and correlations: [tables](tables/). Correlations and scatterplots describe associations, not causal marketing effects.

## Method and reproducibility
Seed 42; a fixed random 80/20 split creates 160 training rows and 40 held-out test rows. All six predeclared candidate pipelines use exactly the same shuffled five-fold training CV splits. Median imputation, scaling, polynomial/interaction transformations, and model fitting occur inside each fold. Candidates use fixed hyperparameters documented in `make_models()`. The lowest mean CV RMSE selects **Quadratic ridge (alpha=1)** before either final held-out evaluation is computed. The held-out set is used once for the selected model and the mean-prediction baseline. Source identifiers never enter the model. No refit on all observations is performed, so reported artifacts retain the split used for evaluation.

| Candidate | CV MAE | CV RMSE ± fold SD | CV R² |
| --- | ---: | ---: | ---: |
| Quadratic ridge (alpha=1) | 0.441 | 0.671 ± 0.375 | 0.977 |
| TV x Radio interaction | 0.701 | 0.968 ± 0.370 | 0.957 |
| Random forest | 0.738 | 1.021 ± 0.247 | 0.956 |
| Linear regression | 1.237 | 1.676 ± 0.297 | 0.870 |
| Ridge (alpha=1) | 1.237 | 1.677 ± 0.289 | 0.871 |
| Dummy mean | 4.191 | 5.150 ± 0.445 | -0.099 |

Fold SD describes variation across these five folds, not a confidence interval. Selection across candidates can make the winning CV score optimistic; the test results are the independent check. See [fold-level results](tables/cross_validation_folds.csv).

## Held-out results
| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Quadratic ridge (alpha=1) | 0.485 | 0.621 | 0.988 |
| Mean baseline | 4.963 | 5.631 | -0.005 |

MAE and RMSE use observed Sales units; lower is better. R² compares squared error with a test-mean reference; higher is better. Predictions for every held-out observation are in [heldout_predictions.csv](heldout_predictions.csv). Residual = observed minus predicted.

![Held-out predictions and residuals](charts/03_heldout_predictions_and_residuals.png)

## Marketing interpretation and hypothetical scenarios
The training-only linear reference model supplies standardized coefficients in [training_linear_associations.csv](tables/training_linear_associations.csv). A coefficient is the fitted Sales association with a one-training-standard-deviation feature increase while holding the other measured variables fixed. It describes the linear reference model, which may differ from the selected nonlinear predictor, and does not establish an intervention effect.

The fitted standardized associations are 3.764 for TV, 2.792 for Radio, and 0.056 for Newspaper, in observed Sales units per training standard deviation of the corresponding measurement. TV and Radio therefore show stronger conditional linear associations than Newspaper in this training sample; a small Newspaper coefficient does not establish that its true causal effect is zero.

![Linear association coefficients](charts/04_linear_association_coefficients.png)

Scenarios start from actual training source row 20 and vary one measured channel by −20% or +20%, clipped to that channel's training range. [Scenario inputs and predictions](tables/hypothetical_scenarios.csv) are hypothetical, with no observed outcome. Each input stays within its marginal training range; joint support is not assured. The results do not estimate causal uplift, guaranteed return on investment, or an optimal budget.

At this reference, predicted Sales is 14.662. The separate +20% input scenarios predict 15.903 for TV, 15.570 for Radio, and 14.681 for Newspaper while holding the other two values fixed. These are sensitivity examples from the selected model, not estimates of what an actual spend intervention will produce.

Business next step: prioritize measured, randomized TV and Radio channel experiments, including an appropriate Newspaper comparison, to test these observational hypotheses. Collect dates, verified spend and sales units, channel costs/revenue, and relevant segment labels before changing budgets. Predefine the experiment's outcome and validation period; this dataset alone does not justify reallocating spend.

![Hypothetical scenarios](charts/05_hypothetical_scenarios.png)

## Limitations and next steps
The small dataset contains no dates, customer segments, platform details, geography, prices, or economic controls. Random held-out accuracy does not establish future-period validity or transport to other markets. Unmeasured confounding, nonlinear structure, and correlated features limit interpretation. A time-stamped dataset and rolling chronological validation are needed for a future forecast; randomized or otherwise defensibly identified interventions are needed for causal marketing decisions. Verify provenance, permission to redistribute, and the measurement units before production use. The original user-supplied CSV is included for assignment reproducibility without an invented upstream license.

## Reproduction
From the project directory, install the pinned dependencies and run `python analysis.py`. The script uses paths relative to its own file and the noninteractive Agg renderer. `python /absolute/path/to/analysis.py` therefore works from another directory too. `--data` and `--output-dir` accept optional alternate paths. The notebook calls the same pipeline and displays its saved tables and charts. Seeds and numeric settings are fixed; package versions can affect final floating-point digits.
