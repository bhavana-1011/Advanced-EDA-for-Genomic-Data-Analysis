import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load encoded features and cluster assignments
df = pd.read_csv("outputs/clusters.csv")

# Reduce dimensions using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(df.iloc[:, :-1])

# Create DataFrame for visualization
pca_df = pd.DataFrame(reduced_features, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["Cluster"]

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="viridis", data=pca_df, alpha=0.7, edgecolors="k")
plt.title("Clustering Visualization (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.savefig("outputs/clustering_visualization.png")  # Save plot
plt.show()

print("✅ Clustering visualization saved to outputs/clustering_visualization.png")
