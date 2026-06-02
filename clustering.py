import pandas as pd
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.patches as mpatches

PARQUET = r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\all_streets.parquet"
TYPES = ["Anti-social behaviour", "Violence and sexual offences"]

# AGGREGATE TO MONTHLY COUNTS PER REGION PER CRIME TYPE
monthly = (
    pl.scan_parquet(PARQUET)
    .filter(pl.col("Crime type").is_in(TYPES))
    .group_by(["Falls within", "year", "month", "Crime type"])
    .agg(pl.len().alias("count"))
    .sort(["Falls within", "year", "month"])
    .collect()
)

monthly = monthly.pivot(
    values="count",
    index=["Falls within", "year", "month"],
    on="Crime type",
    aggregate_function="sum"
).rename({
    "Anti-social behaviour": "asb_count",
    "Violence and sexual offences": "violence_count",
    "Falls within": "region"
}).fill_null(0)

# EXTRACT FEATURES PER REGION
def extract_features(region_df, crime_col):
    # MEAN
    counts = region_df[crime_col].to_numpy()
    months = region_df["month"].to_numpy()

    mean = counts.mean()

    # TREND
    x = np.arange(len(counts))
    trend = np.polyfit(x, counts, 1)[0]

    # SEASONALITY
    month_avgs = [counts[months == m].mean() for m in range(1, 13)
                  if len(counts[months == m]) > 0]
    seasonality = np.std(month_avgs)

    # COVID-DROP
    pre_covid = region_df.filter(
        (pl.col("year") < 2020)
    )[crime_col].mean()

    covid_period = region_df.filter(
        (pl.col("year") == 2020) &
        (pl.col("month") >= 4) &
        (pl.col("month") <= 9)
    )[crime_col].mean()

    if pre_covid and covid_period:
        covid_drop = (covid_period - pre_covid) / pre_covid
    else:
        covid_drop = 0

    return mean, trend, seasonality, covid_drop

# BUILD FEATURE MATRIX
rows = []

for region in monthly["region"].unique().to_list():
    region_df = monthly.filter(pl.col("region") == region).sort(["year", "month"])

    asb_mean, asb_trend, asb_seas, asb_covid    = extract_features(region_df, "asb_count")
    viol_mean, viol_trend, viol_seas, viol_covid = extract_features(region_df, "violence_count")

    rows.append({
        "region":           region,
        "asb_mean":         asb_mean,
        "asb_trend":        asb_trend,
        "asb_seasonality":  asb_seas,
        "asb_covid_drop":   asb_covid,
        "viol_mean":        viol_mean,
        "viol_trend":       viol_trend,
        "viol_seasonality": viol_seas,
        "viol_covid_drop":  viol_covid,
    })

feature_df = pd.DataFrame(rows).set_index("region")

feature_cols = [
    "asb_mean", "asb_trend", "asb_seasonality", "asb_covid_drop",
    "viol_mean", "viol_trend", "viol_seasonality", "viol_covid_drop"
]

# ELBOW PLOT
X_all = StandardScaler().fit_transform(feature_df[feature_cols])

inertias = []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow plot")
plt.tight_layout()
plt.show()

# CLUSTERING
specialist = [
    "British Transport Police",
    "City of London Police",
    "Police Service of Northern Ireland"
]
feature_df_clean = feature_df[~feature_df.index.isin(specialist)].copy()

# IDENTICAL CLUSTERS
feature_df_clean = feature_df_clean.sort_index()
X = StandardScaler().fit_transform(feature_df_clean[feature_cols])

km = KMeans(n_clusters=5, random_state=42, n_init=10)
feature_df_clean["cluster"] = km.fit_predict(X)

# BUILD LABEL MAP MANUALLY
met_cluster     = feature_df_clean.loc["Metropolitan Police Service", "cluster"]
gmp_cluster     = feature_df_clean.loc["Greater Manchester Police", "cluster"]
wm_cluster      = feature_df_clean.loc["West Midlands Police", "cluster"]
cumbria_cluster = feature_df_clean.loc["Cumbria Constabulary", "cluster"]
large_cluster   = (set(range(5)) - {met_cluster, gmp_cluster, wm_cluster, cumbria_cluster}).pop()

cluster_info = {
    cumbria_cluster: ("#4CAF50", "Cluster A — Small/rural forces"),
    met_cluster:     ("#F44336", "Cluster B — Metropolitan Police"),
    gmp_cluster:     ("#FF9800", "Cluster C — Greater Manchester"),
    large_cluster:   ("#2196F3", "Cluster D — Large regional forces"),
    wm_cluster:      ("#9C27B0", "Cluster E — Major urban forces"),
}

# PRINT CLUSTER INFORMATION
for c, (_, label) in cluster_info.items():
    regions = feature_df_clean[feature_df_clean["cluster"] == c].index.tolist()
    print(f"\n{label} ({len(regions)} regions):")
    print(regions)

print("\nCluster feature means:")
print(feature_df_clean.groupby("cluster")[feature_cols].mean().round(2))

# LOAD GEOJSON
gdf = gpd.read_file(r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\PFA_DEC_2024_EW_BGC.geojson")

# NAME MAPPING
name_mapping = {
    "Metropolitan Police":  "Metropolitan Police Service",
    "Cumbria":              "Cumbria Constabulary",
    "Lancashire":           "Lancashire Constabulary",
    "Merseyside":           "Merseyside Police",
    "Greater Manchester":   "Greater Manchester Police",
    "Cheshire":             "Cheshire Constabulary",
    "Northumbria":          "Northumbria Police",
    "Durham":               "Durham Constabulary",
    "North Yorkshire":      "North Yorkshire Police",
    "West Yorkshire":       "West Yorkshire Police",
    "South Yorkshire":      "South Yorkshire Police",
    "Humberside":           "Humberside Police",
    "Cleveland":            "Cleveland Police",
    "West Midlands":        "West Midlands Police",
    "Staffordshire":        "Staffordshire Police",
    "West Mercia":          "West Mercia Police",
    "Warwickshire":         "Warwickshire Police",
    "Derbyshire":           "Derbyshire Constabulary",
    "Nottinghamshire":      "Nottinghamshire Police",
    "Lincolnshire":         "Lincolnshire Police",
    "Leicestershire":       "Leicestershire Police",
    "Northamptonshire":     "Northamptonshire Police",
    "Cambridgeshire":       "Cambridgeshire Constabulary",
    "Norfolk":              "Norfolk Constabulary",
    "Suffolk":              "Suffolk Constabulary",
    "Bedfordshire":         "Bedfordshire Police",
    "Hertfordshire":        "Hertfordshire Constabulary",
    "Essex":                "Essex Police",
    "Thames Valley":        "Thames Valley Police",
    "Hampshire":            "Hampshire Constabulary",
    "Surrey":               "Surrey Police",
    "Kent":                 "Kent Police",
    "Sussex":               "Sussex Police",
    "London, City of":      "City of London Police",
    "Devon & Cornwall":     "Devon & Cornwall Police",
    "Avon and Somerset":    "Avon and Somerset Constabulary",
    "Gloucestershire":      "Gloucestershire Constabulary",
    "Wiltshire":            "Wiltshire Police",
    "Dorset":               "Dorset Police",
    "North Wales":          "North Wales Police",
    "Gwent":                "Gwent Police",
    "South Wales":          "South Wales Police",
    "Dyfed-Powys":          "Dyfed-Powys Police",
}

# JOIN CLUSTERS TO MAP
gdf["region"] = gdf["PFA24NM"].map(name_mapping)
cluster_labels = feature_df_clean[["cluster"]].reset_index()
gdf = gdf.merge(cluster_labels, on="region", how="left")

gdf["color"] = gdf["cluster"].map(
    lambda x: cluster_info[x][0] if pd.notna(x) else "#cccccc"
)

# CHOROPLETH MAP
fig, ax = plt.subplots(figsize=(10, 12))
gdf.plot(color=gdf["color"], ax=ax, edgecolor="white", linewidth=0.5)

patches = [mpatches.Patch(color=v[0], label=v[1]) for v in cluster_info.values()]
patches.append(mpatches.Patch(color="#cccccc", label="Excluded"))
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)

ax.set_title("UK Police Regions by Crime Forecast Cluster",
             fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("cluster_map.png", dpi=150, bbox_inches="tight")
plt.show()

# SAVE CLUSTERS
feature_df_clean[["cluster"]].to_csv(r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\cluster_assignments.csv")
