import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# CONFIG

PARQUET = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"

FORECAST_MONTHS = 12

CRIME_CONFIG = {
    "Anti-social behaviour": {
        "changepoint_prior_scale": 0.01,
        "yearly_seasonality": 10,
    },
    "Violence and sexual offences": {
        "changepoint_prior_scale": 0.05,
        "yearly_seasonality": 10,
    },
}

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

# COVID HOLIDAY

COVID_EVENTS = pd.DataFrame([
    {"holiday": "covid_lockdown", "ds": pd.Timestamp("2020-03-01"), "lower_window": 0, "upper_window": 12}
])

# DATA PREP

def load_monthly(crime_type: str) -> pd.DataFrame:
    raw = (
        pl.scan_parquet(PARQUET)
        .filter(pl.col("Crime type") == crime_type)
        .group_by(["year", "month"])
        .agg(pl.len().alias("count"))
        .sort(["year", "month"])
        .collect()
    )
    df = raw.to_pandas()
    df["ds"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df.rename(columns={"count": "y"})[["ds", "y"]]

# PROPHET FORECAST

def fit_and_forecast(df: pd.DataFrame, crime_type: str, periods: int = FORECAST_MONTHS):
    cfg = CRIME_CONFIG[crime_type]
    model = Prophet(
        yearly_seasonality=cfg["yearly_seasonality"],
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=cfg["changepoint_prior_scale"],
        interval_width=0.95,
        holidays=COVID_EVENTS,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return model, forecast

# PLOTTING

def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame, crime_type: str):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["ds"], df["y"],
            label="Actual", color="steelblue", linewidth=2, zorder=3)
    ax.plot(forecast["ds"], forecast["yhat"],
            label="Forecast", color="tomato", linewidth=2, linestyle="--")
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2, color="tomato", label="95% CI"
    )

    cutoff = df["ds"].max()
    ax.axvspan(cutoff, forecast["ds"].max(), alpha=0.06, color="grey")
    ax.axvline(cutoff, color="grey", linestyle=":", linewidth=1.2)
    ax.axvspan(
        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-03-01"),
        alpha=0.08, color="orange", label="COVID period"
    )

    ax.set_title(f"Crime Forecast: {crime_type}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Incidents")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(df: pd.DataFrame, model: Prophet, crime_type: str):
    """Run cross-validation, print metrics, and plot MAPE by horizon."""
    print(f"\n{'='*55}")
    print(f"  Evaluating: {crime_type}")
    print(f"{'='*55}")

    cv = cross_validation(
        model,
        initial="1825 days",
        period="180 days",
        horizon="365 days"
    )

    metrics = performance_metrics(cv)
    print(metrics[["horizon", "mae", "mape", "rmse"]])
    print(f"  AVG MAPE             : {metrics['mape'].mean():.4f}")

    # MAPE excluding COVID period
    cv_filtered = cv[
        (cv["ds"] < pd.Timestamp("2020-03-01")) |
        (cv["ds"] > pd.Timestamp("2020-11-01"))
    ]
    metrics_filtered = performance_metrics(cv_filtered)
    print(f"  MAPE (excl. COVID)   : {metrics_filtered['mape'].mean():.4f}")

    # MAPE by horizon plot
    metrics["horizon_days"] = metrics["horizon"].dt.days
    plt.plot(metrics["horizon_days"], metrics["mape"])
    plt.title(f"MAPE by Forecast Horizon: {crime_type}", fontsize=12, fontweight="bold")
    plt.xlabel("Horizon (days)")
    plt.ylabel("MAPE")
    plt.xticks([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365])
    plt.tight_layout()
    plt.show()

# MAIN

for crime in CRIME_CONFIG:
    print(f"\nProcessing: {crime} …")
    df = load_monthly(crime)
    model, forecast = fit_and_forecast(df, crime)
    plot_forecast(df, forecast, crime)

    out_path = f"forecast_{crime.lower().replace(' ', '_').replace('/', '_')}.csv"
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_path, index=False)
    print(f"  Forecast saved → {out_path}")

    # Refit for evaluation (cross_validation needs a fresh unfitted model internally)
    df = load_monthly(crime)
    model, _ = fit_and_forecast(df, crime)
    evaluate_model(df, model, crime)
