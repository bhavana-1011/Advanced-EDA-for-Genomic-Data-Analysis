import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load clustered data
df = pd.read_csv("outputs/clusters.csv")

# Extract features and cluster labels
X = df.iloc[:, :-1].values  # All except last column
labels = df["Cluster"].values

# Compute metrics
silhouette = silhouette_score(X, labels)
db_index = davies_bouldin_score(X, labels)

print(f"✅ Silhouette Score: {silhouette:.4f} (Higher is better)")
print(f"✅ Davies-Bouldin Index: {db_index:.4f} (Lower is better)")
