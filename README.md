# 🚗 ft_linear_regression

A tiny yet complete linear regression toolkit: train a model on a CSV, visualize the fit, and interactively predict values.

## ✨ Features

### 📚 Core

- ⌨️ CLI with two commands: `train` and `predict`
- 🧮 Simple hypothesis: $\hat{y} = \theta_0 + \theta_1 x$
- 📦 CSV dataset loader with header support and named columns
- 📉 Batch gradient descent with per-feature standardization for stable training
- 💾 Model save/load to JSON (mirrors dataset name in `models/`)
- 📈 Beautiful plot of data points and the regression line (save to PNG and/or show)
- ⏱️ Progress bar during training

### 🎁 Nice-to-haves

- Clean, minimal codebase with small reusable building blocks (`Vector`, `Dataset`, `FeatureScaler`, `LinearRegressionModel`)
- Makefile tasks for common workflows (venv, train, predict, tests, coverage)

## 📦 Prerequisites

- Python 3.13
- macOS/Linux/Windows

All Python dependencies are pinned in `requirements.txt`. The Makefile will create and use a local virtual environment.

## 🛠️ Setup

```sh
make venv
```

Optional sanity checks:

```sh
make test
make coverage
```

## 🚀 Quickstart

Train on the sample dataset and visualize the result:

```sh
make train
```

This will:

- Train a model on `datasets/data.csv`
- Save it to `models/data.json`
- Open a plot window if `--plot` is provided in the Makefile command (already included)

Predict interactively using the saved model:

```sh
make predict
```

You’ll get a prompt. Type a number (feature value) to see the predicted target. Press Ctrl+D to exit.

## 🧭 Usage (CLI)

The CLI is provided by `ft_linear_regression.py`.

### 📚 Train

```sh
.venv/bin/python ft_linear_regression.py train -d datasets/data.csv
```

What it does:

- Loads a two-column CSV (or more) using header names
- Selects the first two columns by default or use `--feature/--target` to pick specific ones
- Standardizes the feature for training, then converts parameters back to original scale
- Saves the model JSON next to your dataset name inside `models/`
- Optionally shows a plot and/or saves it as a PNG
- Prints training statistics if `--statistics` is provided

### 🔍 Predict

```sh
.venv/bin/python ft_linear_regression.py predict -m models/data.json
```

Enter values at the prompt to see predictions. If the model file is missing, a zero-initialized model is used and you’ll see a warning.

## 💡 Tips

- Your CSV must have a header row; by default the first two columns are used.
- Use `--feature` and `--target` to pick specific columns by name.
- Use `--save-plot` without a value to save next to the model file (same stem, `.png`).
- Press Ctrl+C or Ctrl+D to exit the predictor.

## 👓 Interpreting training statistics

After training with the `--statistics` flag the program prints a few common metrics. Here is a quick, non-technical guide to what each one means and how to read them:

- MSE (Mean Squared Error): the average of squared prediction errors. Because errors are squared, this value emphasises larger mistakes and is expressed in the squared units of the target.
- RMSE (Root Mean Squared Error): the square root of MSE. It is in the same units as the target and can be read as a "typical" prediction error size.
- MAE (Mean Absolute Error): the average of absolute errors. Less sensitive to large outliers than RMSE and also in the target's units.
- R² (Coefficient of Determination): the fraction of the target's variance that the model explains. Values closer to 1 mean the model captures more of the variability; values near 0 mean it captures little. R² can be negative in pathological cases (model worse than predicting the mean).

Quick tips for understanding the numbers:
- Compare RMSE or MAE to the typical size of your target (for example divide by the mean target) to get a relative error — percentages are easier to judge than raw units.
- If RMSE is much larger than MAE, a few large errors (outliers) are likely inflating the score.
- R² gives a sense of explanatory power but not error magnitude; high R² with large RMSE means predictions follow the trend but still miss by a lot in absolute terms.
- Always inspect residuals (errors) visually: residual vs. feature or residual histogram can reveal non-linearity, heteroscedasticity, or outliers.

These metrics give a compact summary of fit quality. Use them together rather than relying on a single number.

## 🗂️ Model storage

Trained models are stored as JSON files under the `models/` directory. Each dataset under `datasets/` maps to a file with the same stem:

- Dataset: `datasets/data.csv`
- Model: `models/data.json`

### 📄 JSON format

```json
{
	"feature_name": "km",
	"target_name": "price",
	"thetas": [13390.42, -85.31]
}
```

- `feature_name` / `target_name`: Copied from the dataset headers
- `thetas`: Trained parameters `[θ0, θ1]` in the original data scale

## 🧪 Testing

```sh
make test
make coverage
```

Unit tests are under `tests/`. Coverage reports are generated to `coverage.xml` and printed in the terminal.

## ⚖️ License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

