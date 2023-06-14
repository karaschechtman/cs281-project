# Fairness Criteria and Accuracy

## Setup
- `pip install -r requirements.txt`: install requirements
- `python scripts/download_data.py`: download COMPAS data into `data/compas.csv`
- `python scripts/predict.py`: train & run logistic regressions (unprocesssed and postprocessed to meet group calibration, thresholdless EO criteria) and save output to `data/compas_predictions.csv`

## Analysis
- `scorer.py`: Computes weighted Brier scores, weighted log loss scores, and beta scores over predictor outputs
- `fairness_utils.py`: Computes various fairness criteria (calibration gap, separation gap) over predictor outputs
- `analysis.ipynb`: Generates main plots (dealing with the relationship of the scoring rule to fairness criteria) for report
- `other_plots.ipynb`: Generates supplementary plots for report