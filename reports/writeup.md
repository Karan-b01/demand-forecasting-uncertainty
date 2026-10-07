# Demand Forecasting with Uncertainty: Method and Results

## Goal and data

The goal is to help a retailer choose inventory while accounting for uncertainty and unequal costs of missed sales and leftover stock. The experiment uses Walmart's M5 daily sales data. To keep modeling practical while retaining varied demand, it samples 100 product IDs across the sales range and keeps all 10 stores. The held-out test set has 28,000 store/item/day records from April 25 through May 22, 2016. Data is split chronologically; training ends March 27, validation covers March 28 through April 24, and the final 28 days are test data.

Features include day, week, month, weekday, weekend, calendar event and price information, sales lags, and rolling statistics. For a forecast on day `t`, sales history features stop at `t - 7`. Rolling values are computed within each store/item series. A rolling 56-day lower-tail sales estimate was added after checking that the pooled P10 model was stuck close to zero for high-volume products.

LightGBM trains a squared-error point model and quantile models for P10, P50, P75, P90, P95, and P99. A validation comparison chooses P10's blend with the rolling 56-day empirical P10 separately for each training-history volume tier. Candidate rolling weights are 0, 0.25, 0.50, 0.75, and 1.00. The saved model selected 0 for low- and medium-volume series and 1 for high-volume series. This raised FOODS_3_090's mean P10 from about 1 to 32.3 units; its test mean sales are 55.0 and mean P50 is 51.3. Its interval width fell from about 79 to 48 units. Blend selection uses validation data only.

## Point forecast and interval results

| Forecast | MAE (units) | RMSE (units) |
| --- | ---: | ---: |
| Seasonal naive (same weekday last week) | 1.368 | 3.402 |
| 28-day moving average | 1.162 | 2.968 |
| LightGBM point forecast | 1.114 | 2.571 |
| LightGBM median (P50) | 1.027 | 2.743 |

P50 has the lowest MAE, while its RMSE is higher than the point model's (2.743 versus 2.571). This is expected: a median forecast minimizes absolute error, while the point model uses squared-error loss and is judged more heavily on large misses. Point accuracy improvements are modest; the main contribution is the range and inventory decision.

The P10-P90 interval targets 80% coverage. On the test set, it covers 89.1% of all records with a mean width of 3.24 units. This overall figure masks an important difference:

| Test records | Count | Coverage | Mean interval width |
| --- | ---: | ---: | ---: |
| All days | 28,000 | 89.1% | 3.24 |
| Zero-sales days | 15,098 | 99.1% | 1.80 |
| Positive-sales days | 12,902 | 77.4% | 4.92 |

Zero demand is common, and a zero lower bound contains those observations. This makes all-day coverage look high even though the model misses too many positive-sales outcomes. Quantile tables show strict-below and at-or-below shares separately: P10 is tied exactly with actuals on 54.0% of records, mostly at zero. Strictly below P10 is 0.9%; at-or-below is 54.8%. These shares are not interval coverage; coverage is directly computed from both interval bounds.

CQR was calibrated separately for low, medium, and high volume tiers using training-history tiers known before each forecast. In this run, each validation correction rounded to 0.000 units, so the calibrated intervals are unchanged. The overall 80% target is already over-covered on validation due to the zero-heavy demand distribution, so marginal CQR does not expand intervals to fix the weaker positive-sales slice. Calibrating only records that are later observed to have positive sales would use an unknown future outcome; a future improvement would need a deployable two-part model that estimates both the chance of any sale and the amount sold conditional on a sale.

## Quantile interval versus residual interval

The bonus residual interval adds fixed validation residual percentiles from the point model. It is narrower overall, but its error pattern differs by demand regime:

| Method / slice | Count | Coverage | Mean width |
| --- | ---: | ---: | ---: |
| Quantile interval, all test days | 28,000 | 89.1% | 3.24 |
| Residual interval, all test days | 28,000 | 79.6% | 2.34 |
| Quantile interval, positive-sales days | 12,902 | 77.4% | 4.92 |
| Residual interval, positive-sales days | 12,902 | 69.6% | 2.58 |
| Quantile interval, demand spikes | 1,835 | 9.6% | 2.87 |
| Residual interval, demand spikes | 1,835 | 29.3% | 2.26 |

The residual interval is about 28% narrower and its overall coverage is closer to the nominal 80%. It is not uniformly better: on high-volume test records its coverage is 57.4%, versus 87.7% for the quantile interval. By volume tier, residual coverage is 95.0%, 86.6%, and 57.4% for low, medium, and high; quantile coverage is 90.1%, 89.4%, and 87.7%. The residual range has constant-width residual offsets and cannot adapt as well to different demand scales. The quantile model is preferable for heterogeneous volume, but neither method is reliable on spikes.

## Inventory decision and cost sensitivity

Let `Cu` be the cost of one missed sale and `Co` the cost of one extra unit. The cost-based stocking percentile is:

```text
tau* = Cu / (Cu + Co)
```

For `Cu = $3` and `Co = $1`, the target is `3 / (3 + 1) = 0.75`, or P75. Purchase quantity is the ceiling of the target stock minus on-hand and inbound stock, floored at zero. P75 is now a directly trained LightGBM quantile rather than an interpolation.

| Cost ratio (Cu:Co) | Target percentile | Lowest grid cost on test set |
| --- | ---: | ---: |
| 1:1 | 50% | $28,750 |
| 2:1 | 66.7% | $45,702 at about P67 |
| 3:1 | 75% | $56,402 at P75 |
| 5:1 | 83.3% | $72,357 at about P82 |

At the 3:1 ratio, point-forecast stocking costs $64,328 on the same test observations, compared with $56,402 for P75: an estimated $7,926 (12.3%) reduction in this simulation. The curve checks a grid from P50 to P99; minima can differ slightly from the theoretical target because of the grid and imperfect quantile forecasts.

![Simulated inventory cost by stocking quantile and underage-to-overage cost ratio](figures/inventory_cost_by_quantile.png)

These are retrospective costs from user-selected penalties, not guaranteed real-world savings. Actual purchase recommendations should also account for lead time, pack sizes, current stock, inbound orders, and supplier constraints.

## Failure analysis and limitations

Spike days are defined as sales more than two historical standard deviations above the 28-day historical mean. The interval covers only 9.6% of 1,835 such records. It does better on usual demand (94.7%), showing that surprise spikes remain the clearest overconfidence case. Holiday/event coverage is 89.6%, close to 89.0% on regular calendar days. The possible-promotion row contains only 51 price-drop records and is inconclusive; a price decrease is a proxy, not a complete promotion label.

The M5 sample includes only 100 of 3,049 products, though it spans all 10 stores and the complete 28-day test window. Quantile coverage is marginal and does not promise conditional coverage, particularly for spikes or positive-only demand. More reliable spike forecasts would need richer promotion/stockout signals and a longer rolling test over more dates and product samples. See `reports/interval_method_comparison.csv` for method-by-slice measurements and `reports/figures/` for generated diagnostics.

## Reproduction

With the M5 CSV files installed as described in `README.md`, run `powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1`. The script prepares data, engineers features, retrains the models, recalibrates, and regenerates evaluation, inventory, and failure-analysis outputs. Tests can be run using `python -m pytest -q`.
