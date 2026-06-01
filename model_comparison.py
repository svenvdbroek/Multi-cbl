import warnings
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import date
import pmdarima as pm
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Config

PARQUET = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"

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

COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2021-03-01")   # matches Prophet upper_window=12

COVID_EVENTS = pd.DataFrame([{
    "holiday":      "covid_lockdown",
    "ds":           pd.Timestamp("2020-03-01"),
    "lower_window": 0,
    "upper_window": 12,
}])

CV_INITIAL = "1825 days"
CV_PERIOD  = "180 days"
CV_HORIZON = "365 days"

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
})

# Data loading

print("Loading data …")

monthly_all = (
    pl.scan_parquet(PARQUET)
    .filter(pl.col("Crime type").is_in(list(CRIME_CONFIG.keys())))
    .group_by(["year", "month", "Crime type"])
    .agg(pl.len().alias("count"))
    .sort(["year", "month"])
    .collect()
)


def to_prophet_df(crime: str) -> pd.DataFrame:
    sub = monthly_all.filter(pl.col("Crime type") == crime)
    df  = sub.to_pandas()
    df["ds"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df.rename(columns={"count": "y"})[["ds", "y"]].sort_values("ds").reset_index(drop=True)


def to_series(crime: str) -> pd.Series:
    df = to_prophet_df(crime)
    return pd.Series(df["y"].values, index=pd.DatetimeIndex(df["ds"]), name=crime)


def make_covid_dummy(idx) -> np.ndarray:
    """Binary 1/0 array: 1 during COVID period. Accepts any index-like input."""
    dti = pd.DatetimeIndex(idx)
    result = (dti >= COVID_START) & (dti <= COVID_END)
    # .values only exists on pandas objects; if already numpy, use directly
    return result.values.astype(float) if hasattr(result, "values") else result.astype(float)


# Prophet CV

def run_prophet_cv(crime: str) -> pd.DataFrame:
    df  = to_prophet_df(crime)
    cfg = CRIME_CONFIG[crime]
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
    cv = cross_validation(model, initial=CV_INITIAL, period=CV_PERIOD,
                          horizon=CV_HORIZON, disable_tqdm=True)
    cv["model"] = "Prophet"
    return cv


# SARIMA rolling CV

def run_sarima_cv(crime: str, cutoffs: pd.Series, use_covid_dummy: bool) -> pd.DataFrame:
    series = to_series(crime)
    label  = "SARIMA+COVID" if use_covid_dummy else "SARIMA"
    records = []

    for cutoff in cutoffs:
        train = series[series.index <= cutoff]
        if len(train) < 24:
            continue

        future_idx = pd.date_range(
            cutoff + pd.DateOffset(months=1), periods=12, freq="MS"
        )

        try:
            if use_covid_dummy:
                exog_train  = make_covid_dummy(train.index).reshape(-1, 1)
                exog_future = make_covid_dummy(future_idx).reshape(-1, 1)

                auto_model = pm.auto_arima(
                    train.values,                  # pass as plain numpy
                    exogenous=exog_train,
                    m=12, seasonal=True, stepwise=True,
                    information_criterion="aic",
                    max_p=2, max_q=2, max_P=1, max_Q=1,
                    error_action="ignore", suppress_warnings=True, trace=False,
                )
                fc_mean = auto_model.predict(
                    n_periods=12,
                    exogenous=exog_future,
                )
            else:
                auto_model = pm.auto_arima(
                    train.values,                  # pass as plain numpy
                    m=12, seasonal=True, stepwise=True,
                    information_criterion="aic",
                    max_p=2, max_q=2, max_P=1, max_Q=1,
                    error_action="ignore", suppress_warnings=True, trace=False,
                )
                fc_mean = auto_model.predict(n_periods=12)

            for ds, yhat in zip(future_idx, fc_mean):
                if ds in series.index:
                    records.append({
                        "ds":     ds,
                        "yhat":   float(yhat),
                        "y":      float(series[ds]),
                        "cutoff": cutoff,
                        "model":  label,
                    })

        except Exception as e:
            import traceback
            print(f"    {label} failed at {cutoff.strftime('%Y-%m')}: {e}")
            traceback.print_exc()
            continue

    return pd.DataFrame(records)


# Metrics

def compute_metrics(cv_df: pd.DataFrame, label: str) -> dict:
    def _mape(a, p):
        return (np.abs((a - p) / np.where(a == 0, np.nan, a))).mean() * 100

    if cv_df is None or cv_df.empty or "y" not in cv_df.columns:
        print(f"    !  No data for {label} — skipping metrics.")
        return {"Model": label, "MAE": float("nan"), "RMSE": float("nan"),
                "Avg MAPE (%)": float("nan"), "Avg MAPE excl COVID": float("nan")}

    actual = cv_df["y"].values
    pred   = cv_df["yhat"].values

    mask_excl = ~((cv_df["ds"] >= COVID_START) & (cv_df["ds"] <= COVID_END))
    a_ex = cv_df.loc[mask_excl, "y"].values
    p_ex = cv_df.loc[mask_excl, "yhat"].values

    return {
        "Model":               label,
        "MAE":                 round(mean_absolute_error(actual, pred), 1),
        "RMSE":                round(np.sqrt(mean_squared_error(actual, pred)), 1),
        "Avg MAPE (%)":        round(_mape(actual, pred), 4),
        "Avg MAPE excl COVID": round(_mape(a_ex, p_ex), 4),
    }


# MAPE by horizon plot

def plot_mape_by_horizon(cv_dict: dict, crime: str) -> None:
    colors = {"Prophet": "#1a9641", "SARIMA": "#d7191c", "SARIMA+COVID": "#f77f00"}
    fig, ax = plt.subplots(figsize=(10, 4))

    for label, cv_df in cv_dict.items():
        if cv_df is None or cv_df.empty or "ds" not in cv_df.columns:
            print(f"    !  Skipping {label} in horizon plot — no data.")
            continue
        df = cv_df.copy()
        df["horizon_days"] = (df["ds"] - df["cutoff"]).dt.days
        df["abs_pct_err"]  = np.abs((df["y"] - df["yhat"]) /
                                     df["y"].replace(0, np.nan)) * 100
        hm = df.groupby("horizon_days")["abs_pct_err"].mean()
        ax.plot(hm.index, hm.values, label=label,
                color=colors.get(label, "grey"), linewidth=2)

    ax.set_title(f"MAPE by Forecast Horizon: {crime}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365])
    ax.legend()
    plt.tight_layout()
    plt.show()


# Overlay plot

def plot_overlay(series: pd.Series, cv_dict: dict, crime: str) -> None:
    colors = {"Prophet": "#1a9641", "SARIMA": "#d7191c", "SARIMA+COVID": "#f77f00"}
    styles = {"Prophet": "-.", "SARIMA": "--", "SARIMA+COVID": (0, (3, 1, 1, 1))}

    valid = {k: v for k, v in cv_dict.items()
             if v is not None and not v.empty and "cutoff" in v.columns}
    if not valid:
        print("    !  No valid CV data for overlay plot.")
        return

    last_cutoff = max(v["cutoff"].max() for v in valid.values())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axvspan(COVID_START, COVID_END, color="#fff3cd", alpha=0.5, label="COVID period", zorder=0)
    ax.axvline(last_cutoff, color="grey", linestyle=":", linewidth=1.2)
    ax.plot(series.index, series.values, label="Actual", color="#2c7bb6", linewidth=1.8, zorder=3)

    for label, cv_df in valid.items():
        last = cv_df[cv_df["cutoff"] == cv_df["cutoff"].max()]
        mape = (np.abs((last["y"] - last["yhat"]) /
                        last["y"].replace(0, np.nan))).mean() * 100
        ax.plot(last["ds"], last["yhat"],
                label=f"{label}",
                color=colors.get(label, "grey"),
                linestyle=styles.get(label, "--"),
                linewidth=2)

    ax.set_title(f"Model Comparison — Last CV Window: {crime}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Incidents")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    plt.show()


# Main loop

summary = []

for crime in CRIME_CONFIG:
    print(f"\n{'═'*60}")
    print(f"  {crime}")
    print(f"{'═'*60}")

    print("  Running Prophet CV …")
    prophet_cv = run_prophet_cv(crime)
    cutoffs    = prophet_cv["cutoff"].drop_duplicates().sort_values()
    print(f"  Cutoffs: {len(cutoffs)}")

    print(f"  Running SARIMA (no dummy) on {len(cutoffs)} cutoffs …")
    sarima_cv = run_sarima_cv(crime, cutoffs, use_covid_dummy=False)

    print(f"  Running SARIMA+COVID (with dummy) on {len(cutoffs)} cutoffs …")
    sarima_covid_cv = run_sarima_cv(crime, cutoffs, use_covid_dummy=True)

    cv_dict = {
        "Prophet":      prophet_cv,
        "SARIMA":       sarima_cv,
        "SARIMA+COVID": sarima_covid_cv,
    }

    print(f"\n  {'Metric':<25} {'Prophet':>12} {'SARIMA':>12} {'SARIMA+COVID':>14}")
    print(f"  {'-'*63}")
    all_metrics = {k: compute_metrics(v, k) for k, v in cv_dict.items()}
    for key in ["MAE", "RMSE", "Avg MAPE (%)", "Avg MAPE excl COVID"]:
        vals = [str(all_metrics[m][key]) for m in ["Prophet", "SARIMA", "SARIMA+COVID"]]
        print(f"  {key:<25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>14}")

    for m in ["Prophet", "SARIMA", "SARIMA+COVID"]:
        row = all_metrics[m]
        row["Crime type"] = crime
        summary.append(row)

    plot_mape_by_horizon(cv_dict, crime)
    plot_overlay(to_series(crime), cv_dict, crime)

# Final summary

print(f"\n{'═'*60}")
print("  FINAL SUMMARY")
print(f"{'═'*60}")
df_out = pd.DataFrame(summary)[["Crime type", "Model", "MAE", "RMSE", "Avg MAPE (%)", "Avg MAPE excl COVID"]]
print(df_out.to_string(index=False))
print("\nDone.")
