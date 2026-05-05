import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
from datetime import date

# MEMORY EFFICIENT DATAFRAME

PARQUET = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"
 
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

df = pl.scan_parquet(PARQUET)

TYPES = ["Anti-social behaviour", "Violence and sexual offences"]

# CRIME COUNT BAR CHART
 
crime_counts = (
    df.group_by("Crime type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .collect()
)
 
plt.figure(figsize=(10, 5))
plt.bar(crime_counts["Crime type"].to_list(), crime_counts["count"].to_list())
plt.title("Distribution of Crime Types")
plt.xlabel("Crime Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
 
# CRIME PERCENTAGE BAR CHART
 
total_crimes = crime_counts["count"].sum()
crime_pct = crime_counts.with_columns(
    (pl.col("count") / total_crimes * 100).alias("percentage")
)
 
plt.figure(figsize=(10, 5))
plt.bar(crime_pct["Crime type"].to_list(), crime_pct["percentage"].to_list())
plt.title("Crime Type Distribution (%)")
plt.xlabel("Crime Type")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
 
# CRIME PER LSOA COUNT
 
lsoa_counts = (
    df.filter(pl.col("LSOA code").is_not_null())
    .group_by("LSOA code")
    .agg(pl.len().alias("total_crimes"))
    .sort("total_crimes", descending=True)
    .collect()
)
pl.Config.set_tbl_rows(20)
print(lsoa_counts.head(20))
 
# NULL VALUE COUNT
 
null_stats = (
    df.select([
        pl.len().alias("total"),
        pl.col("Crime ID").is_null().sum().alias("nulls"),
    ])
    .collect()
)
 
total = null_stats["total"][0]
nulls = null_stats["nulls"][0]
print("Total rows :", total)
print("Null values:", nulls)
print("Null %     :", round(nulls / total * 100, 2), "%")

# CRIME THROUGHOUT MONTHS

monthly = (
    pl.scan_parquet(PARQUET)
    .filter(pl.col("Crime type").is_in(TYPES))
    .group_by(["year", "month", "Crime type"])
    .agg(pl.len().alias("count"))
    .sort(["year", "month"])
    .collect()
)

fig, ax = plt.subplots(figsize=(12, 5))

for crime in TYPES:
    sub = monthly.filter(pl.col("Crime type") == crime)
    dates = [date(y, m, 1) for y, m in zip(sub["year"], sub["month"])]
    ax.plot(dates, sub["count"], label=crime, linewidth=2)

ax.set_title("Monthly Crime Counts")
ax.legend()
ax.set_ylim(0, 300000)
plt.tight_layout()
plt.show()

# OUTCOME DISTRIBUTION

outcomes = (
    df.filter(pl.col("Last outcome category").is_not_null())
    .group_by(["Crime type", "Last outcome category"])
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .collect()
)

for crime in TYPES:
    sub = outcomes.filter(pl.col("Crime type") == crime)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sub["Last outcome category"].to_list()[::-1],
            sub["count"].to_list()[::-1])
    ax.set_title(f"Outcome Distribution: {crime}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of incidents")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

# Null value count for outcome in Anti-social behaviour crime type

anti_social = (
    df.filter(pl.col("Crime type") == "Anti-social behaviour")
    .select([
        pl.len().alias("total"),
        pl.col("Last outcome category").is_null().sum().alias("nulls"),
    ])
    .collect()
)

total = anti_social["total"][0]
nulls = anti_social["nulls"][0]
print(f"Null %: {round(nulls / total * 100, 2)}%  ({nulls:,} / {total:,})")

# MONTHLY CRIME COUNT

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

monthly_totals = (
    df.group_by(["Crime type", "month"])
    .agg(pl.len().alias("count"))
    .sort("month")
    .collect()
)

for crime in TYPES:
    sub = monthly_totals.filter(pl.col("Crime type") == crime)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([month_names[m - 1] for m in sub["month"].to_list()], sub["count"].to_list())
    ax.set_title(f"Total Crimes per Month: {crime}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of incidents")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
