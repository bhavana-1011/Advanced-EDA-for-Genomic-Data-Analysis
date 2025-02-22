import torch
import pandas as pd
import pickle
from autoencoder import load_trained_autoencoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load preprocessed data (4 features)
data_path = "outputs/processed_vcf_data.csv"
df = pd.read_csv(data_path).select_dtypes(include=["number"])

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

# Load trained autoencoder
best_model_path = "outputs/best_autoencoder.pth"
input_dim = data_tensor.shape[1]  # Should be 4 now
autoencoder = load_trained_autoencoder(best_model_path, input_dim, encoding_dim=2)

# Extract encoded features
encoded_features = autoencoder.encode(data_tensor).detach().numpy()

# Perform clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(encoded_features)

# Save KMeans model
with open("outputs/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Save encoded features & cluster labels
encoded_df = pd.DataFrame(encoded_features)
encoded_df["Cluster"] = kmeans.labels_
encoded_df.to_csv("outputs/clusters.csv", index=False)

print("✅ Encoded features & clusters saved for evaluation!")
print("✅ KMeans model saved successfully!")
