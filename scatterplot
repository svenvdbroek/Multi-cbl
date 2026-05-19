import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PARQUET = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"
DEPRIVATION_CSV = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\deprivation.csv"

TYPES = ["Anti-social behaviour", "Violence and sexual offences"]


plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

# CRIME RATE PER LSOA
lsoa_crimes = (
    pl.scan_parquet(PARQUET)
    .filter(pl.col("LSOA code").is_not_null())
    .group_by("LSOA code")
    .agg(pl.len().alias("crime_count"))
    .collect()
)

# DEPRIVATION DATA
deprivation = pl.read_csv(
    DEPRIVATION_CSV,
    separator=",",
    infer_schema_length=5000,
)

# TAKE NEEDED COLUMNS
deprivation = deprivation.select([
    pl.col("LSOA code (2011)").alias("LSOA code"),
    pl.col("Index of Multiple Deprivation (IMD) Score").alias("imd_score"),
    pl.col("Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)").alias("imd_decile"),
])

# JOIN & COMPUTE CRIME RATE
merged = lsoa_crimes.join(deprivation, on="LSOA code", how="inner")

# SCATTERPLOT: IMD score vs crime rate
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    merged["imd_score"].to_list(),
    merged["crime_count"].to_list(),
    alpha=0.3,
    s=10,
    color="steelblue",
)

# TRENDLINE
x = merged["imd_score"].to_numpy()
y = merged["crime_count"].to_numpy()
mask = np.isfinite(x) & np.isfinite(y)
m, b = np.polyfit(x[mask], y[mask], 1)
ax.plot(sorted(x[mask]), [m * xi + b for xi in sorted(x[mask])],
        color="crimson", linewidth=1.5, label=f"Trend (slope={m:.2f})")

ax.set_title("Deprivation vs Crime Count per LSOA", fontsize=13, fontweight="bold")
ax.set_xlabel("IMD Score (higher = more deprived)")
ax.set_ylabel("Total crimes recorded")
ax.set_yscale("log")
ax.legend()
plt.tight_layout()
plt.show()

# FULL DATASET STATS
pearson_r, pearson_p = stats.pearsonr(merged["imd_score"], merged["crime_count"])
spearman_r, spearman_p = stats.spearmanr(merged["imd_score"], merged["crime_count"])

print(f"Full_Pearson  r={pearson_r:.3f}, p={pearson_p:.4f}")
print(f"Full_Spearman r={spearman_r:.3f}, p={spearman_p:.4f}")

# PER CRIME TYPE: PLOT + STATS
for crime in TYPES:
    lsoa_crimes_type = (
        pl.scan_parquet(PARQUET)
        .filter(pl.col("LSOA code").is_not_null())
        .filter(pl.col("Crime type") == crime)
        .group_by("LSOA code")
        .agg(pl.len().alias("crime_count"))
        .collect()
    )

    merged_type = lsoa_crimes_type.join(deprivation, on="LSOA code", how="inner")

    x = merged_type["imd_score"].to_numpy()
    y = merged_type["crime_count"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[mask], y[mask], 1)

    # PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.3, s=10, color="steelblue")
    ax.plot(sorted(x[mask]), [m * xi + b for xi in sorted(x[mask])],
            color="crimson", linewidth=1.5, label=f"Trend (slope={m:.2f})")
    ax.set_title(f"Deprivation vs Crime Count: {crime}", fontsize=13, fontweight="bold")
    ax.set_xlabel("IMD Score (higher = more deprived)")
    ax.set_ylabel("Total crimes recorded")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # STATS
    pearson_r, pearson_p = stats.pearsonr(x[mask], y[mask])
    spearman_r, spearman_p = stats.spearmanr(x[mask], y[mask])
    print(f"\n{crime}")
    print(f"Pearson  r={pearson_r:.3f}, p={pearson_p:.4f}")
    print(f"Spearman r={spearman_r:.3f}, p={spearman_p:.4f}")
