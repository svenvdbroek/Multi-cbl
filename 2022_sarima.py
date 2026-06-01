import warnings
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

warnings.filterwarnings("ignore")

# Config

PARQUET = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"

TYPES = ["Anti-social behaviour", "Violence and sexual offences"]

FORECAST_MONTHS = 12   # how many months ahead to predict
TEST_MONTHS     = 6    # hold out last 6 months

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
})

# Load & aggregate (only from after covid)

print("Loading data (2022+) …")

monthly = (
    pl.scan_parquet(PARQUET)
    .filter(
        pl.col("Crime type").is_in(TYPES) &
        (pl.col("year") >= 2022)
    )
    .group_by(["year", "month", "Crime type"])
    .agg(pl.len().alias("count"))
    .sort(["year", "month"])
    .collect()
)


def to_series(crime: str) -> pd.Series:
    """Return a monthly Pandas Series with a proper DatetimeIndex."""
    sub = monthly.filter(pl.col("Crime type") == crime)
    idx = pd.to_datetime([
        date(y, m, 1) for y, m in zip(sub["year"].to_list(), sub["month"].to_list())
    ])
    s = pd.Series(sub["count"].to_list(), index=idx, name=crime)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full_idx, fill_value=0)


# Stationarity check (ADF)

def adf_report(series: pd.Series) -> None:
    result = adfuller(series.dropna())
    print(f"\n  ADF statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Stationary    : {'Yes' if result[1] < 0.05 else 'No — auto_arima will handle differencing'}")


# Diagnostic plots (ACF / PACF)

def plot_diagnostics(series: pd.Series, crime: str) -> None:
    # With ~30 months, keep lags to 24 max to avoid overfitting the plot
    max_lags = min(24, len(series) // 2 - 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf( series.diff().dropna(), lags=max_lags, ax=axes[0], title=f"ACF — {crime} (2022+)")
    plot_pacf(series.diff().dropna(), lags=max_lags, ax=axes[1], title=f"PACF — {crime} (2022+)")
    plt.tight_layout()
    plt.show()


# Auto-fit SARIMA & forecast

def fit_and_forecast(series: pd.Series, crime: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  Crime type   : {crime}")
    print(f"  Observations : {len(series)}  ({series.index[0].strftime('%Y-%m')} → {series.index[-1].strftime('%Y-%m')})")

    if len(series) < 24:
        print(" ! Fewer than 24 observations — SARIMA results may be unreliable.")

    print("  Stationarity test (raw series):")
    adf_report(series)

    print("  Plotting ACF / PACF …")
    plot_diagnostics(series, crime)

    # Train / test split
    train = series.iloc[:-TEST_MONTHS]
    test  = series.iloc[-TEST_MONTHS:]

    print(f"\n  Train : {train.index[0].strftime('%Y-%m')} → {train.index[-1].strftime('%Y-%m')}  ({len(train)} months)")
    print(f"  Test  : {test.index[0].strftime('%Y-%m')} → {test.index[-1].strftime('%Y-%m')}  ({len(test)} months)")

    # Fit on train only
    print("\n  Running auto_arima on train set …")
    auto_model = pm.auto_arima(
        train,
        m=12,
        seasonal=True,
        stepwise=True,
        information_criterion="aic",
        max_p=2, max_q=2,      # constrain search space given limited data
        max_P=1, max_Q=1,
        error_action="ignore",
        suppress_warnings=True,
        trace=True,
    )

    print(f"\n   Best order    : {auto_model.order}")
    print(f"   Best seasonal : {auto_model.seasonal_order}")
    print(auto_model.summary())

    result = auto_model.arima_res_

    # Out of sample prediction over the test period 
    test_forecast_obj  = result.get_forecast(steps=TEST_MONTHS)
    test_forecast_mean = test_forecast_obj.predicted_mean
    test_conf_int      = test_forecast_obj.conf_int()

    # Out of sample MAPE
    mape_oos = (abs((test.values - test_forecast_mean.values) / test.values)).mean() * 100
    mae_oos  = mean_absolute_error(test.values, test_forecast_mean.values)
    rmse_oos = np.sqrt(mean_squared_error(test.values, test_forecast_mean.values))

    print(f"\n  Out-of-sample metrics (last {TEST_MONTHS} months):")
    print(f"    MAPE : {mape_oos:.2f}%   ← based on held-out test data")
    print(f"    MAE  : {mae_oos:,.1f}")
    print(f"    RMSE : {rmse_oos:,.1f}")

    # Plot train/test evaluation
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train.index, train, label="Train (observed)", color="#2c7bb6", linewidth=1.8)
    ax.plot(test.index,  test,  label="Test (observed)",  color="#fdae61", linewidth=1.8)
    ax.plot(test_forecast_mean.index, test_forecast_mean,
            label=f"Test forecast  |  MAPE: {mape_oos:.1f}%",
            color="#d7191c", linewidth=2, linestyle="--")
    ax.fill_between(
        test_conf_int.index,
        test_conf_int.iloc[:, 0],
        test_conf_int.iloc[:, 1],
        color="#d7191c", alpha=0.15, label="95% CI",
    )

    order_str = f"SARIMA{auto_model.order}×{auto_model.seasonal_order}"
    ax.set_title(f"{order_str} — Train/Test Evaluation (2022+): {crime}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of incidents")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Refit on full 2022+ series, then forecast into the future
    print("\n  Refitting on full 2022+ series for future forecast …")
    auto_model.update(test)
    result_full = auto_model.arima_res_

    fitted_full     = pd.Series(result_full.fittedvalues, index=series.index)
    future_obj      = result_full.get_forecast(steps=FORECAST_MONTHS)
    future_idx      = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=FORECAST_MONTHS, freq="MS")
    future_mean     = pd.Series(future_obj.predicted_mean, index=future_idx)
    future_conf_int = pd.DataFrame(future_obj.conf_int(), index=future_idx)

    # Plot full 2022+ series + future forecast 
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(series.index, series, label="Observed (2022+)", color="#2c7bb6", linewidth=1.8)
    ax.plot(fitted_full.index, fitted_full, label="Fitted", color="#abd9e9",
            linewidth=1.2, linestyle="--", alpha=0.8)
    ax.plot(future_mean.index, future_mean,
            label=f"Forecast (+{FORECAST_MONTHS} months)",
            color="#d7191c", linewidth=2)
    ax.fill_between(
        future_conf_int.index,
        future_conf_int.iloc[:, 0],
        future_conf_int.iloc[:, 1],
        color="#d7191c", alpha=0.15, label="95% CI",
    )

    ax.set_title(f"{order_str} — Future Forecast (2022+): {crime}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of incidents")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Future forecast table
    fc_df = pd.DataFrame({
        "Month":    future_mean.index.strftime("%Y-%m"),
        "Forecast": future_mean.round(0).astype(int).values,
        "Lower CI": future_conf_int.iloc[:, 0].round(0).astype(int).values,
        "Upper CI": future_conf_int.iloc[:, 1].round(0).astype(int).values,
    })
    print(f"\n  {crime} — {FORECAST_MONTHS}-month future forecast:")
    print(fc_df.to_string(index=False))


# Main

for crime_type in TYPES:
    series = to_series(crime_type)
    fit_and_forecast(series, crime_type)

print("\nDone.")
