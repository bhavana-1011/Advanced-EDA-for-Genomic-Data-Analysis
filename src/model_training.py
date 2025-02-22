import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from autoencoder import Autoencoder

# Load the preprocessed data
data_path = "outputs/processed_vcf_data.csv"
df = pd.read_csv(data_path)

# Select only numeric columns (4 features)
df = df.select_dtypes(include=[np.number])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Convert to PyTorch tensor
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

# Hyperparameter tuning options
hyperparams = {
    "encoding_dim": [2],  # Reduce encoding size
    "learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [16, 32, 64]
}

best_loss = float('inf')
best_params = {}

# K-Fold Cross Validation (3 folds for speed)
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Iterate over different hyperparameters
for encoding_dim in hyperparams["encoding_dim"]:
    for lr in hyperparams["learning_rate"]:
        for batch_size in hyperparams["batch_size"]:
            fold_losses = []

            print(f"\nTraining with Encoding Dim: {encoding_dim}, Learning Rate: {lr}, Batch Size: {batch_size}")

            for train_idx, val_idx in kf.split(data_tensor):
                train_data = data_tensor[train_idx]
                val_data = data_tensor[val_idx]

                # Define the autoencoder
                input_dim = train_data.shape[1]
                autoencoder = Autoencoder(input_dim, encoding_dim)

                # Loss and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

                # Training loop
                epochs = 15  # Reduced for speed
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = autoencoder(train_data)
                    loss = criterion(outputs, train_data)
                    loss.backward()
                    optimizer.step()

                # Validate the model
                val_outputs = autoencoder(val_data)
                val_loss = criterion(val_outputs, val_data).item()
                fold_losses.append(val_loss)

            avg_val_loss = np.mean(fold_losses)
            print(f"Avg Validation Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {"encoding_dim": encoding_dim, "learning_rate": lr, "batch_size": batch_size}
                torch.save(autoencoder.state_dict(), "outputs/best_autoencoder.pth")

print("\nBest Parameters:", best_params)
print("✅ Hyperparameter tuning & cross-validation complete!")
