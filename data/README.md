# M5 data setup

The raw M5 Forecasting dataset is not included in this repository. Download it directly from the [M5 Forecasting competition on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and follow Kaggle's terms for access, use, and redistribution.

Place the following files in `data/raw/`:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

The current preparation pipeline reads those three files. Other competition files, including `sales_train_validation.csv` and submission templates, are not required by this implementation. Do not commit the downloaded files, derived full-size datasets, or model artifacts. They are ignored by `.gitignore`.

From the repository root, create the directory if needed:

```powershell
New-Item -ItemType Directory -Force data/raw
```

After placing the files, follow the reproduction steps in the root `README.md`.
