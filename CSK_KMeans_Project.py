import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.decomposition
from sklearn.metrics import silhouette_score

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "CSK_200_Raw_Synthetic.csv", na_values=["NA"])

features = ["csk_runs", "opponent_runs", "csk_wickets",
            "opponent_wickets", "csk_sixes"]

print("\n1. DATASET SELECTION AND HANDLING")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.dtypes)
print("\nFirst 10 Records")
print(df.head(10))
assert len(df) == 200

print("\n2. DATA PRE-PROCESSING")
print("Missing values before handling:")
print(df.isnull().sum())
clean = df.copy()
for col in features:
    clean[col] = clean[col].fillna(clean[col].median())
print("Missing values after handling:")
print(clean.isnull().sum())

for col in features:
    q1 = clean[col].quantile(0.25)
    q3 = clean[col].quantile(0.75)
    iqr = q3 - q1
    clean[col] = clean[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

scaler = StandardScaler()
X = scaler.fit_transform(clean[features])
print("\n3. EXPLORATORY DATA ANALYSIS")
print(clean[features].describe(percentiles=[0.10,0.25,0.50,0.75,0.90]))
print("Mean:\n", clean[features].mean())
print("Variance:\n", clean[features].var())
print("Skewness:\n", clean[features].skew())
print("Kurtosis:\n", clean[features].kurt())

print("\n4. K-MEANS CLUSTERING")
k_values = range(2, 9)
scores = []
for k in k_values:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    scores.append(silhouette_score(X, labels))
best_k = list(k_values)[int(np.argmax(scores))]
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clean["cluster"] = model.fit_predict(X)
print("Best number of clusters:", best_k)

print("\n5. MODEL EVALUATION")
score = silhouette_score(X, clean["cluster"])
print("Silhouette score:", round(score, 4))
print(clean["cluster"].value_counts().sort_index())
print(clean.groupby("cluster")[features].mean().round(2))
clean.to_csv(folder / "CSK_200_Processed_Clusters.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].hist(clean["csk_runs"], bins=14, edgecolor="black")
axes[0, 0].set_title("CSK Runs Histogram")
axes[0, 0].set_xlabel("Runs")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].boxplot([clean["csk_runs"], clean["opponent_runs"]])
axes[0, 1].set_xticks([1, 2])
axes[0, 1].set_xticklabels(["CSK", "Opponent"])
axes[0, 1].set_title("Runs Boxplot")
axes[0, 1].set_ylabel("Runs")

inertias = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
axes[1, 0].plot(range(1, 9), inertias, marker="o")
axes[1, 0].set_title("K-Means Elbow Method")
axes[1, 0].set_xlabel("Number of Clusters")
axes[1, 0].set_ylabel("WCSS")

points = sklearn.decomposition.PCA(n_components=2).fit_transform(X)
scatter = axes[1, 1].scatter(points[:, 0], points[:, 1],
                              c=clean["cluster"], cmap="viridis", s=40)
axes[1, 1].set_title("K-Means Cluster Visualization")
axes[1, 1].set_xlabel("Principal Component 1")
axes[1, 1].set_ylabel("Principal Component 2")
fig.colorbar(scatter, ax=axes[1, 1], label="Cluster")

plt.tight_layout()
graph_path = folder / "CSK_KMeans_4_Graphs.png"
plt.savefig(graph_path, dpi=180, bbox_inches="tight")
print("Saved graphs to:", graph_path)
plt.show()