# Movie Recommendation System

A student-built recommendation project with multiple collaborative filtering models, hybrid modeling, evaluation scripts, and a Streamlit app.

## Team
- Soham Shrivastava
- Vansh Kumar
- Pranav Somase
- Arin Mehta

## Project Overview
This project compares different recommendation approaches on movie ratings data and reports both:
- Rating prediction quality (RMSE, MAE)
- Ranking quality (Precision@10, Recall@10, Diversity@10, Novelty@10, SerendipityProxy@10)

Implemented models:
- Baseline bias model
- Matrix Factorization (MF) with optional temporal decay
- SVD++ (implicit feedback)
- KNN collaborative filtering
- Hybrid recommender (MF + content similarity)

## Repository Structure
- `src/` - Data loading and preprocessing utilities
- `models/` - Recommender model implementations
- `evaluation/` - Metrics and evaluation logic
- `app/` - Streamlit application
- `outputs/` - Saved CSV results and model artifacts
- `report/` - LaTeX report and paper material
- `evaluate_ranking.py` - Ranking evaluation entrypoint
- `tune_hyperparameters.py` - Hyperparameter tuning entrypoint
- `Training&Evaluation.ipynb` - Notebook workflow

## Requirements
Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main packages:
- pandas
- numpy
- scikit-learn
- streamlit
- tqdm
- plotly

## Setup
From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

### 1) Hyperparameter Tuning (MF + Hybrid alpha)

```bash
python tune_hyperparameters.py
```

Output:
- `outputs/tuning_results.csv`

### 2) Ranking Evaluation

Quick sanity run (fast):

```bash
python evaluate_ranking.py --quick
```

Medium/full run example:

```bash
python evaluate_ranking.py --max-users 1000 --candidate-sample-size 200 --mf-epochs 5 --svdpp-epochs 1 --svdpp-max-implicit-items 50
```

If you want faster runs, skip SVD++:

```bash
python evaluate_ranking.py --max-users 1000 --candidate-sample-size 200 --mf-epochs 5 --skip-svdpp
```

Output:
- `outputs/ranking_results.csv`

### 3) RMSE/MAE Evaluation
Use notebook or evaluation utilities (from `evaluation/evaluate.py`) to produce:
- `outputs/results.csv`

### 4) Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Notes On Runtime
- SVD++ is the slowest model.
- Use `--quick` for fast validation.
- `tqdm` progress bars are enabled in tuning/evaluation loops.
- Candidate sampling and user caps reduce ranking evaluation runtime significantly.

## Current Best Observations (from latest runs)
- MF performs best on RMSE/MAE in current setup.
- SVD++ performs best on Precision@10 and Recall@10.
- Hybrid currently needs more tuning for stronger ranking performance.

## Reproducibility Tips
- Keep train/test split and random seeds fixed (`random_state=42` in scripts).
- Compare models under the same evaluation settings (`k`, user cap, candidate sample size).
- Use quick mode only for sanity checks, not final comparison tables.

## Report
LaTeX report is available at:
- `report/report.tex`

Compile with:

```bash
pdflatex report/report.tex
pdflatex report/report.tex
```

## Troubleshooting
1. If `.pkl` files or `__pycache__` appear in Git status:
- They may be tracked already; remove from Git index once using `git rm --cached ...`.

2. If ranking run seems stuck:
- It is usually in SVD++ fit stage.
- Run with `--skip-svdpp` or lower `--svdpp-epochs` / set `--svdpp-max-implicit-items`.

3. If runtime is too high:
- Lower `--max-users`
- Lower `--candidate-sample-size`
- Lower `--train-frac`
- Use `--quick`
