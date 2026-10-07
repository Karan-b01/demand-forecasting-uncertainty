# Demand Forecasting Project Guide

This guide explains the project from the business question to the model, dashboard, and results. It is written for a reader who may be new to forecasting and inventory mathematics.

## 1. What problem does the project solve?

A shop needs to decide how much of each product to have available. If it orders too little, it may miss sales. If it orders too much, it may pay to store or discount leftovers. A single forecast such as “sell 10 units” does not show how uncertain that number is.

This project predicts several possible levels of daily demand, displays a likely range, and uses the cost of running short versus having leftovers to recommend a stock target. The main question is:

> Given the past sales and the business's costs, what quantity should we plan to have available, and how uncertain is that recommendation?

The project uses Walmart's public M5 Forecasting data from Kaggle. It is a historical research and demonstration tool. It does **not** produce live forecasts for future dates.

## 2. What happens, from data to recommendation?

```mermaid
flowchart LR
    A[Download M5 files from Kaggle] --> B[Prepare a daily product-store table]
    B --> C[Create calendar, sales-history and price features]
    C --> D[Train point and quantile LightGBM models]
    D --> E[Calibrate prediction ranges on validation data]
    E --> F[Evaluate once on later test dates]
    F --> G[FastAPI sends saved results to React dashboard]
    G --> H[Choose a stock target from shortage and leftover costs]
```

The project keeps the data in date order:

1. **Training period:** the model learns relationships between past sales and later demand.
2. **Validation period:** model choices and interval adjustments are selected here.
3. **Test period:** the final 28 days are held aside to measure performance on dates the model did not train or calibrate on.

The current experiment samples 100 products across different demand levels and includes all 10 M5 stores. The test set has 28,000 product-store-day records, covering April 25 through May 22, 2016. It is a useful project-sized sample, not the entire M5 catalog.

## 3. Data and features

The preparation script reads these Kaggle files from `data/raw/`:

- `sales_train_evaluation.csv`: daily unit sales by product and store.
- `calendar.csv`: dates, weekdays, and M5 event information.
- `sell_prices.csv`: weekly product prices by store.

The original Kaggle files are not included in this repository. See [`data/README.md`](../data/README.md) for download instructions and Kaggle terms.

For each product and store, the feature pipeline builds:

| Feature group | Examples | Why it helps |
| --- | --- | --- |
| Calendar | Weekday, day, month, week number, weekend, month end | Demand can change with the day or time of year. |
| Events | M5 event name, event type, event-day indicator | Holidays and named events can affect shopping. |
| Sales lags | Sales from 7, 14, 21, 28, and 35 days earlier | Recent and same-weekday history can be informative. |
| Rolling sales | Recent mean, standard deviation, minimum and maximum over 7, 14, and 28 days | Summarizes the recent level and variability of demand. |
| Lower-tail history | Rolling 10th-percentile sales over 28 and 56 days | Helps estimate low-demand days, especially for high-volume products. |
| Price | Lagged price and recent price ratio | A price drop can hint at a promotion, but it is only a proxy. |
| Product/store identity | Item, store, department, category and state | Lets the model learn differences between products and locations. |

### Avoiding a look-ahead mistake

The forecast uses a seven-day horizon/cutoff. For a target day called `t`, sales-based features stop at `t - 7` or earlier. In plain English, the model is not allowed to use sales from the days it is supposed to predict. The leakage tests check this behavior.

## 4. The forecasting model

The project uses **LightGBM**, a tree-based machine-learning model that learns patterns from structured tables. One model predicts an ordinary point forecast, and separate quantile models predict different places in the demand distribution.

### What do P10, P50, P75 and P90 mean?

- **P10:** a lower-demand estimate. Roughly 10% of comparable outcomes should be below it when the quantile is calibrated.
- **P50:** the median, or middle estimate. Half of comparable outcomes should be below it.
- **P75:** a higher-than-middle estimate, used for the example cost-based inventory target.
- **P90:** a cautious upper estimate. Roughly 90% of comparable outcomes should be below it when calibrated.

The model also produces P95 and P99 for the inventory-cost curve. P10 through P90 make an **80% nominal prediction interval**: the band from the 10th percentile to the 90th percentile.

### Quantile model formula

For a chosen percentile `tau`, quantile regression minimizes the pinball loss:

```text
error = actual demand - predicted quantile
loss  = max(tau * error, (tau - 1) * error)
```

**Easy English:** this loss scores an estimate according to the percentile it is meant to represent. It treats overestimates and underestimates differently, so P10, P50, and P90 learn different parts of the demand range instead of all predicting the average.

### Special care for the lower forecast

Many product-store pairs have zero sales on many days. In a model trained across many products, that can pull P10 too close to zero even for busy products. The project compares the LightGBM P10 with a 56-day historical lower-demand estimate on validation data, then selects a blend by demand-volume tier. This decision is made without looking at test outcomes. For the saved experiment, high-volume series use the historical lower-tail estimate; low- and medium-volume series use the model's P10. For example, the FOODS_3_090 item's average P10 increased from about 1 to about 32 units after this change.

## 5. Prediction ranges and calibration

A model's 80% label is a target, not a guarantee. We check how often actual test demand falls between the lower and upper predictions.

### Coverage formula

```text
Coverage = number of actuals inside the interval / number of test records
```

An actual is inside when:

```text
lower forecast <= actual demand <= upper forecast
```

For an 80% interval, coverage near 80% is the overall goal. The average width is:

```text
Average width = average(upper forecast - lower forecast)
```

**Easy English:** coverage tells us how often the range contains the true sales. Width tells us how wide the range usually is. A very wide range may cover many outcomes but be less useful for planning, so both numbers matter.

### Conformalized Quantile Regression (CQR)

CQR checks how far validation actuals fell outside the raw P10-to-P90 range. For each validation record `i`, it calculates:

```text
score_i = max(P10_i - actual_i, actual_i - P90_i)
```

It then takes a high percentile of those scores, called `q_hat`, and adjusts the test limits:

```text
calibrated lower = max(0, P10 - q_hat)
calibrated upper = max(0, P90 + q_hat)
```

**Easy English:** if validation demand often falls outside the raw band, CQR uses those misses to widen the later band. Here it is calculated separately for low-, medium-, and high-volume groups, whose group assignments come from training-period sales.

In this saved run, all three CQR adjustments were `0.000` units. So CQR did not change the ranges. This is a limitation of the current method on this zero-heavy data: the many zero-demand outcomes make overall calibration look better than coverage on selling days. The adjustment does not fix every subgroup just because the overall statistic looks acceptable.

### How the project checks quantiles and intervals

The project reports both the **strictly below** share (`actual < forecast`) and the **at or below** share (`actual <= forecast`). This distinction matters when actual sales equal a forecast exactly. Around 54% of records tie the P10 forecast, mostly because both sales and P10 are zero. Those ties make the inclusive P10 share much larger than its 10% target. Quantile shares and P10-to-P90 interval coverage are related checks, but they are not the same calculation.

The project also reports the **Winkler score**:

```text
Winkler score = interval width + (2 / alpha) * distance outside the interval
```

Here `alpha = 0.20` for an 80% range. A miss is penalized according to how far it is outside the band. **Easy English:** this score rewards ranges that are narrow, but gives a strong penalty when they miss. Lower is better.

## 6. How forecast accuracy is measured

The model is compared with simple baselines: repeating sales from the same weekday last week and using a 28-day historical average.

### MAE: average miss

```text
MAE = average(abs(actual demand - forecast))
```

**Easy English:** ignore whether the model was too high or too low, measure the size of each miss, and average those sizes. Lower is better.

### RMSE: gives large misses extra weight

```text
RMSE = sqrt(average((actual demand - forecast)^2))
```

**Easy English:** square each miss before averaging, then take the square root. A few large errors count more than they do under MAE. Lower is better.

P50 has the best MAE in the reported comparison, while the point forecast has the best RMSE. This is expected because a median is naturally suited to minimizing absolute error, while the point model is trained for squared-error performance.

## 7. Turning forecasts into an inventory decision

The dashboard asks the user for two business costs:

- **Cu (underage cost):** estimated cost of being short by one unit, such as a lost sale.
- **Co (overage cost):** estimated cost of having one extra unsold unit, such as storage or markdown cost.

The target stock percentile is:

```text
target percentile = Cu / (Cu + Co)
```

**Easy English:** if missing a sale is expensive compared with having leftovers, choose a higher demand percentile and stock more. If leftovers are more expensive, choose a lower percentile and stock less.

Example:

```text
Cu = $3, Co = $1
target = 3 / (3 + 1) = 0.75 = 75%
```

So the dashboard uses P75 as its cost-based stock target for these costs. The selected item's recommended quantity changes as the cost settings change.

When current stock and inbound stock are provided, the purchase quantity is:

```text
buy = ceil(max(0, target stock - on-hand stock - already-on-order stock))
```

**Easy English:** subtract units already available or arriving soon; never recommend a negative purchase; round up because products are bought in whole units.

The dashboard can produce a date-specific purchase list for all products and stores, accept an optional inventory CSV, and download the resulting list as CSV. It also plots a cost curve over stock percentiles so the user can compare the trade-offs rather than seeing only one choice.

### Simulated cost formula

For each test record:

```text
missing units = max(0, actual demand - stock)
extra units   = max(0, stock - actual demand)
record cost   = Cu * missing units + Co * extra units
total cost    = sum(record cost over the test set)
```

**Easy English:** charge the shortage cost for every unit demand exceeded stock, and the leftover cost for every stocked unit that demand did not use. Add those costs across all test days and product-store pairs.

These costs are user-selected assumptions in a retrospective simulation. They are not measured Walmart accounting costs and are not guaranteed real-world savings. The formula does not model supplier lead times, case-pack sizes, storage limits, or every operational constraint.

## 8. Project features and technology

### Forecast and evaluation pipeline

- **Python** organizes data, builds features, trains models, calibrates intervals, and produces evaluation reports.
- **pandas and NumPy** prepare and calculate with the sales tables.
- **LightGBM** trains the point forecast and quantile models.
- **scikit-learn** supplies common accuracy metrics.
- **Matplotlib** generates evaluation and inventory-cost charts.
- **pytest** runs checks for interval calculations and feature leakage.

### Web application

- **FastAPI** serves saved forecast results and inventory recommendations through API endpoints.
- **React** builds the interactive dashboard.
- **Vite** runs and builds the web front end.
- **Recharts** draws the forecast and cost charts; **Lucide** supplies interface icons.
- Cost controls update the recommended stock and cost curve.
- A 28-day replay shows actual sales day by day and tracks whether demand was inside the range.
- The dashboard includes point-accuracy metrics, interval coverage and width, calibration details, failure-analysis groups, cost comparisons, a daily purchase list, and a plain-language glossary.

## 9. Results from the current saved experiment

The figures below come from 28,000 test records: 100 products across 10 stores over the final 28 held-out days.

| Forecast | MAE | RMSE |
| --- | ---: | ---: |
| Same weekday last week | 1.368 | 3.402 |
| 28-day average | 1.162 | 2.968 |
| LightGBM point forecast | 1.114 | 2.571 |
| LightGBM P50 median | 1.027 | 2.743 |

| Interval group | Records | P10-P90 coverage | Average width |
| --- | ---: | ---: | ---: |
| All test records | 28,000 | 89.1% | 3.24 units |
| Zero-sales days | 15,098 | 99.1% | 1.80 units |
| Positive-sales days | 12,902 | 77.4% | 4.92 units |

The intended interval coverage is 80%. Overall coverage is above 80%, but zero-sales days account for over half the records and are easy to include when the lower limit is zero. Positive-day coverage is below target and deserves attention.

For a 3:1 shortage-to-leftover cost ratio, the historical simulation reports:

| Stocking approach | Simulated total cost |
| --- | ---: |
| Point forecast | $64,328 |
| Cost-based P75 | $56,402 |
| Difference | $7,926 lower for P75 (12.3%) |

This is evidence from the chosen historical sample and assumed costs, not a promise of future savings.

## 10. Strongest parts of the project

1. **It represents uncertainty.** The result is a set of demand quantiles and an interval, not only a single number.
2. **The evaluation follows time order.** Later dates are held out from model fitting, reducing the risk of testing on information the model already saw.
3. **The inventory recommendation has a business interpretation.** The selected percentile responds to the relative cost of missed sales and extra stock.
4. **It checks failure groups.** The results reveal that zero-sales days hide poor performance on spikes and positive-sales days.
5. **It is reproducible without committing licensed data.** The code and instructions are in the repository; users download M5 directly from Kaggle and regenerate the results locally.
6. **It is usable as a demonstration.** The React dashboard combines charts, tables, cost settings, replay, and a daily purchase list.

## 11. Weakest parts and limitations

1. **Demand spikes are badly undercovered.** Only 9.6% of 1,835 spike records fall inside the range. This is the clearest case where the model is overconfident. Promotions and sudden events need better signals.
2. **Positive-sales coverage is low.** It is 77.4%, below the nominal 80% target, even though all-record coverage is 89.1%. The zero-heavy average can make the headline look reassuring.
3. **CQR made no change in this run.** Its volume-tier corrections were all zero. It therefore did not repair the spike or positive-day problem.
4. **The test is still limited.** It covers 28 dates and 100 of the 3,049 M5 products, though all 10 stores and varied product demand are represented. More rolling test windows and more items would make the evidence stronger.
5. **Promotion results are inconclusive.** The price-drop proxy has only 51 records. M5 price changes do not identify every promotion.
6. **The point forecast gain is modest.** P50 improves MAE over the simple 28-day average only slightly. The stronger project contribution is the uncertainty-aware inventory decision and its analysis.
7. **The cost savings are simulated.** Costs were selected for the exercise; real purchasing decisions also need lead time, supplier, pack-size, shelf-life, and current-inventory constraints.
8. **This is a historical dashboard.** It shows saved test forecasts and should not be treated as a live replenishment service.

The bonus residual-based interval gets 79.6% coverage at a narrower 2.34-unit average width, compared with 89.1% coverage and 3.24 units for the quantile interval. It is closer to the overall 80% target and narrower on average, but it performs worse on positive days and demand scale can vary by product. Neither approach handles spikes well.

## 12. How to install and run it

First clone/download the repository, install Python dependencies, and download the required Kaggle data as described in the root [README](../README.md) and [`data/README.md`](../data/README.md). Run these commands from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
```

Then start the API in one terminal from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the Vite address shown in the terminal, usually `http://localhost:5173`. Port `8000` is the API, not the dashboard page. To run the tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The reproduction script rebuilds the local data and model outputs; the dashboard cannot use those outputs until the pipeline has run successfully. See the [shorter research-style write-up](writeup.md) and the root [README](../README.md) for setup and repository instructions.

## 13. Plain-language glossary

| Term | Meaning |
| --- | --- |
| Actual demand | Units customers bought in the historical data. |
| Forecast | The model's estimate of future demand based on available history. |
| Quantile | A point in the demand distribution, such as the median or 90th percentile. |
| Prediction interval | Lower and upper estimates intended to contain a stated share of actual outcomes. |
| Calibration | Checking whether forecast percentiles and intervals achieve their intended rates on held-out data. |
| Coverage | Share of actual values that fell inside a prediction interval. |
| Width | Distance between the interval's upper and lower limits. |
| CQR | A validation-based method that uses past interval misses to adjust later ranges. |
| Lag | A value from an earlier date, such as sales seven days ago. |
| Rolling statistic | A summary, such as average or standard deviation, over a recent moving time window. |
| Stockout | Demand cannot be met because available inventory is too low. |
| Leftover / overstock | Inventory remains after the demand for that period. |
| Cu | Estimated cost for each unit of unmet demand. |
| Co | Estimated cost for each extra unit left over. |
| Service target | The chosen share of similar demand outcomes the stock plan aims to cover. |
| Baseline | A simple forecast used as a reference to judge the more complex model. |
| Leakage | Accidentally letting the model use information from after the forecast cutoff. |
