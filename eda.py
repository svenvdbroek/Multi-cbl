import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\svenv\OneDrive\Documenten\Multi_dbl\data\2026-02\2026-02\2026-02-avon-and-somerset-street.csv")

#CRIME COUNT BAR CHART

crime_counts = df['Crime type'].value_counts()

plt.figure(figsize=(12,6))
crime_counts.plot(kind='bar')
plt.title("Distribution of Crime Types")
plt.xlabel("Crime Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#CRIME PERCENTAGE BAR CHART

crime_pct = (df["Crime type"].value_counts(normalize=True) * 100)

crime_pct.plot(kind="bar")

plt.ylabel("Percentage (%)")
plt.xlabel("Crime type")
plt.title("Crime Type Distribution (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

#CRIME PER LSOA COUNT

lsoa_counts = (
    df["LSOA code"]
    .dropna()
    .value_counts()
    .reset_index()
)

lsoa_counts.columns = ["lsoa", "total_crimes"]

print(lsoa_counts.head(20))

#NULL VALUE COUNT

total = len(df)
nulls = df["Crime ID"].isna().sum()
print('Total rows:', total)
print('Null Values:', nulls)
print('Null percentage', round((nulls / total) * 100, 2), '%')
