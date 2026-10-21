# Sales Prediction

Predict sales from TV, radio, and newspaper advertising measurements using Python regression models. Includes data cleaning, model comparison, held-out evaluation, marketing insights, an executed notebook, and reports. The results describe associations in the supplied data rather than causal effects or time-based forecasts.

## Run

Python 3.11+ is recommended. From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

The script uses file-relative paths, Matplotlib's Agg backend, and seed 42. It can also be called from another directory. Optional arguments:

```bash
python analysis.py --data data/Advertising.csv --output-dir outputs
```

Open `analysis.ipynb` with a Jupyter kernel using the same environment. Optional reproducible notebook execution tooling is pinned in `requirements-notebook.txt`: install it with `python -m pip install -r requirements-notebook.txt`. The core CLI needs only `requirements.txt`. The notebook calls the same implementation and displays the results.

## Deliverables

- [Design report and results](outputs/summary.md): objective, data, method, results, interpretation, limitations, and reproduction.
- [Machine-readable metrics and audit](outputs/metrics.json): checksum, data audit, source-row split, CV scores, and held-out metrics.
- [Analysis notebook](analysis.ipynb) and [pipeline](analysis.py).
- [Held-out predictions](outputs/heldout_predictions.csv), [tables](outputs/tables/), and [charts](outputs/charts/).

## Evaluation design

The verified sequential `Unnamed: 0` identifier is excluded from training. After data auditing, an untouched 20% test partition is reserved. Six fixed candidates are compared using five-fold CV on the remaining 80%: a mean baseline, linear regression, ridge, a TV×Radio interaction model, degree-two ridge, and random forest. Imputation, scaling, and feature construction happen inside pipelines within training folds. Mean CV RMSE selects the final model before test evaluation. MAE, RMSE, and R² are reported for the selected model and baseline on the held-out partition.

No dates are present: this is conditional out-of-sample prediction, not an actual future time-series forecast. There are also no customer-segment or online-platform fields, so these are not fabricated. Linear coefficients describe association. Hypothetical scenarios vary observed inputs within training ranges and do not identify causal effects, guaranteed ROI, or an optimal advertising budget.

## Data provenance and units

`data/Advertising.csv` came from the user's uploaded archive. The task document links to [Advertising.csv on Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv); that link is a task-document reference, not an independently verified licensing statement. The supplied file has 200 rows; the initial audit found no missing measurements and no duplicate observations after ignoring the row identifier. The repository retains the supplied data for assignment reproducibility. The original upstream source, redistribution license, currency, and sales-volume units were not independently verified; no upstream licensing claim is made. All numerical results use observed dataset units. Verify data permission and measurement meaning before reuse beyond this assignment.

This repository covers the technical project. Submission forms and any personal social-media/video requirements remain separate actions.

## Report exports

- [Formatted design report](REPORT.md)
- [PDF report](report/report.pdf)
- [Editable DOCX report](report/report.docx)

The committed notebook contains outputs from a successful full execution. The included GitHub Actions workflow reproduces the analysis on Python 3.12. View the Actions tab for the current GitHub run status.

## Submission downloads

- [Complete four-project submission package](https://raw.githubusercontent.com/nisa3131/CodeAlpha_IrisFlowerClassification/main/submission/CodeAlpha_DataScience_Submission.zip)
- [Combined PDF report](https://raw.githubusercontent.com/nisa3131/CodeAlpha_IrisFlowerClassification/main/submission/CodeAlpha_DataScience_Design_Report.pdf) · [Editable Word report](https://raw.githubusercontent.com/nisa3131/CodeAlpha_IrisFlowerClassification/main/submission/CodeAlpha_DataScience_Design_Report.docx)
- [Submission checklist](https://raw.githubusercontent.com/nisa3131/CodeAlpha_IrisFlowerClassification/main/submission/SUBMISSION_CHECKLIST.pdf) · [Video explanation scripts](https://raw.githubusercontent.com/nisa3131/CodeAlpha_IrisFlowerClassification/main/submission/VIDEO_EXPLANATION_SCRIPTS.md)
- [Download this project as a ZIP](https://github.com/nisa3131/CodeAlpha_SalesPrediction/archive/refs/heads/main.zip)

The complete package includes the four projects, executed notebooks, datasets, results, charts, individual reports, the combined report, and submission guidance. The shared downloads are hosted in the Iris repository's `submission/` folder. Record/post the required explanation videos and complete the official submission form separately.
