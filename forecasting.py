import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import os

# CONFIG

PARQUET    = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"
CLUSTERS   = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\cluster_assignments.csv"
OUTPUT = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\cluster_forecasts"

FORECAST_MONTHS = 24

CRIME_CONFIG = {
    "Anti-social behaviour": {
        "changepoint_prior_scale": 0.01,
        "yearly_seasonality": 10,
        "col": "asb_count",
    },
    "Violence and sexual offences": {
        "changepoint_prior_scale": 0.05,
        "yearly_seasonality": 10,
        "col": "violence_count",
    },
}

CLUSTER_LABELS = {
    "A": "Small/rural forces",
    "B": "Metropolitan Police",
    "C": "Greater Manchester",
    "D": "Large regional forces",
    "E": "Major urban forces",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
})

COVID_EVENTS = pd.DataFrame([{
    "holiday":      "covid_lockdown",
    "ds":           pd.Timestamp("2020-03-01"),
    "lower_window": 0,
    "upper_window": 12,
}])

os.makedirs(OUTPUT, exist_ok=True)

# LOAD CLUSTERS

assignments = pd.read_csv(CLUSTERS, index_col="region")

# Map numeric cluster IDs
cluster_id_to_label = {}
anchor_map = {
    "Metropolitan Police Service": "B",
    "Greater Manchester Police":   "C",
    "West Midlands Police":        "E",
    "Cumbria Constabulary":        "A",
}
used_ids = set()
for region, letter in anchor_map.items():
    cid = assignments.loc[region, "cluster"]
    cluster_id_to_label[cid] = letter
    used_ids.add(cid)

remaining_id     = (set(assignments["cluster"].unique()) - used_ids).pop()
cluster_id_to_label[remaining_id] = "D"

assignments["label"] = assignments["cluster"].map(cluster_id_to_label)
print("Cluster label mapping:")
print(assignments["label"].value_counts().sort_index())

# DATA PREP

# Load and aggregate monthly counts per region
monthly = (
    pl.scan_parquet(PARQUET)
    .filter(pl.col("Crime type").is_in(list(CRIME_CONFIG.keys())))
    .group_by(["Falls within", "year", "month", "Crime type"])
    .agg(pl.len().alias("count"))
    .sort(["Falls within", "year", "month"])
    .collect()
    .pivot(
        values="count",
        index=["Falls within", "year", "month"],
        on="Crime type",
        aggregate_function="sum",
    )
    .rename({
        "Anti-social behaviour":        "asb_count",
        "Violence and sexual offences": "violence_count",
        "Falls within":                 "region",
    })
    .fill_null(0)
    .to_pandas()
)

monthly["ds"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" +
    monthly["month"].astype(str).str.zfill(2) + "-01"
)

def load_cluster_series(label: str, crime_col: str) -> pd.DataFrame:
    regions = assignments[assignments["label"] == label].index.tolist()
    cluster_data = (
        monthly[monthly["region"].isin(regions)]
        .groupby("ds")[crime_col]
        .sum()
        .reset_index()
        .rename(columns={crime_col: "y"})
        .sort_values("ds")
    )

    # Trim leading rows where data is effectively zero (mainly used for violence as data starts later)
    first_valid = cluster_data[cluster_data["y"] > 100].index[0]
    cluster_data = cluster_data.loc[first_valid:].reset_index(drop=True)

    # Cluster C (GMP): data disappears after 2020 due to known recording failure
    if label == "C":
        cluster_data = cluster_data[cluster_data["ds"] < "2020-01-01"]

    return cluster_data

# PROPHET 

def fit_and_forecast(df: pd.DataFrame, crime_type: str, label: str) -> tuple:
    cfg = CRIME_CONFIG[crime_type]
    
    # Calculate months needed to reach a common end date for all clusters
    target_end = pd.Timestamp("2028-01-01")
    months_needed = (
        (target_end.year - df["ds"].max().year) * 12 +
        (target_end.month - df["ds"].max().month)
    )

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
    future   = model.make_future_dataframe(periods=months_needed, freq="MS")
    forecast = model.predict(future)
    return model, forecast

# PLOTTING

def plot_forecast(df, forecast, crime_type, cluster_label):
    cluster_name = CLUSTER_LABELS[cluster_label]
    note = "\n Data cut off at Jan 2020 due to GMP recording failure" if cluster_label == "C" else ""

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
    ax.axvspan(cutoff, forecast["ds"].max(),
               alpha=0.06, color="grey")
    ax.axvline(cutoff, color="grey", linestyle=":", linewidth=1.2)
    ax.axvspan(
        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-03-01"),
        alpha=0.08, color="orange", label="COVID period"
    )

    ax.set_title(
        f"Crime Forecast: {crime_type}\nCluster {cluster_label} — {cluster_name}{note}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Incidents")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    fname = f"cluster_{cluster_label}_{crime_type.lower().replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(os.path.join(OUTPUT, fname), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {fname}")

# MAIN

for label in sorted(CLUSTER_LABELS.keys()):
    cluster_name = CLUSTER_LABELS[label]
    print(f"  Cluster {label} — {cluster_name}")

    for crime_type, cfg in CRIME_CONFIG.items():
        print(f"  Fitting Prophet for: {crime_type} …")

        df       = load_cluster_series(label, cfg["col"])
        _, forecast = fit_and_forecast(df, crime_type, label)
        plot_forecast(df, forecast, crime_type, label)

        # Save forecast CSV
        csv_name = f"cluster_{label}_{crime_type.lower().replace(' ', '_').replace('/', '_')}.csv"
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
            os.path.join(OUTPUT, csv_name), index=False
        )
