# DiaLens — Diabetes Health Indicators ML Project

DiaLens is a student machine-learning project that classifies a respondent's **survey-reported** diabetes/prediabetes status from public health indicators. It is a retrospective classification demonstration, not a medical screening tool, diagnosis, or prediction of future disease.

## Project question

Can 21 self-reported health, lifestyle, healthcare-access, and demographic indicators classify whether a respondent reported diabetes or prediabetes in the CDC's 2015 BRFSS survey?

## Dataset

- **Name:** Diabetes Health Indicators Dataset (binary-label BRFSS 2015 file)
- **Size:** 253,680 survey records, 21 predictors, and one binary target
- **Target:** `Diabetes_binary` — 0 means no diabetes reported; 1 combines prediabetes and diabetes
- **Source page:** [Kaggle: Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Underlying survey documentation:** [CDC BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html)
- The included CSV is a mirror of the Kaggle binary file. Keep the Kaggle and CDC attributions in presentations and derivative work.

This dataset is self-reported, from the United States, and from 2015. Its binary positive label groups prediabetes and diabetes together. It should not be described as a global or current clinical dataset.

## Selected machine-learning model

The training workflow evaluates three candidates on a stratified validation split, then keeps one final model for the website. **Histogram Gradient Boosting** was selected because it had the highest validation PR-AUC in this project run. The validation PR-AUC values were 0.4003 for Logistic Regression, 0.4188 for Random Forest, and 0.4327 for Histogram Gradient Boosting.

The threshold is tuned on validation data for positive-class F1. The final metrics below are from a separate stratified 20% test set that was not used to choose the model or threshold. Your results may vary slightly with dependency versions or code changes.

| Held-out test metric | Result |
| --- | ---: |
| Accuracy | 80.69% |
| Positive-class precision | 37.93% |
| Positive-class recall | 60.62% |
| Positive-class F1 | 0.467 |
| Macro-F1 | 0.674 |
| Balanced accuracy | 72.28% |
| Specificity | 83.94% |
| ROC-AUC | 0.827 |
| PR-AUC | 0.423 |

Confusion matrix in `[TN, FP; FN, TP]` order: `[[36655, 7012], [2784, 4285]]`.

Interpret metrics together. In particular, accuracy alone is not enough because the positive class is the minority. The feature chart uses permutation importance on held-out rows; it describes model reliance, not causal effects.

## Run the project

### Recommended on this computer: GitHub Codespaces

This computer's Smart App Control blocked SciPy's compiled extension inside the local virtual environment. Codespaces runs the same Python project in a separate cloud development environment, so the local Windows block does not apply. Create a private GitHub repository, add this project, then use **Code → Codespaces → Create codespace**. Do not upload the existing `.venv` folder or `__pycache__`; `.gitignore` excludes them when using Git. The Codespace automatically installs dependencies and trains the model. After that completes, start the site:

```bash
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open the forwarded port **5000** in the browser. Keep the port private; don't enter real or identifiable health information.

### Local run (only if Windows allows the SciPy package)

Use Python 3.12, then run these separately from this folder:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe train_model.py
.\.venv312\Scripts\python.exe app.py
```

Visit `http://127.0.0.1:5000`. The model and report are generated as `brfss_model.pkl` and `model_report.json`.

## Project files

- `app.py` — Flask website and prediction/batch-scoring endpoints
- `ml_pipeline.py` — shared dataset and model definitions
- `train_model.py` — stratified model selection, threshold tuning, held-out test evaluation, and permutation importance
- `templates/home.html` — responsive DiaLens website
- `diabetes_binary_health_indicators_BRFSS2015.csv` — large binary-label dataset
- `model_report.json` — generated metrics and charts' data
- `.devcontainer/devcontainer.json` — Codespaces setup and private port forwarding
- `PRESENTATION_GUIDE.md` — four-member roles, six-minute run-of-show, and screenshot checklist
- `legacy/` — original small starter dataset and model, archived and unused

## Limitations

The model classifies an existing self-reported survey label; it does not estimate future diabetes or clinical risk. Scores are not calibrated medical probabilities. Survey responses can be inaccurate or outdated, and the data may not represent other locations or groups. Do not use outputs for health decisions and do not upload identifiable health data. This project does not claim clinical readiness or regulatory compliance.
