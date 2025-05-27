# PCA with Scikit-learn (for baseline comparison)
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assume X is (N_samples, L*L) Ising configurations
# Load Ising data
X = np.load("ising_configurations.npy")  # shape: (N, L*L)
T = np.load("temperatures.npy")  # shape: (N,)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA projection colored by temperature
plt.figure(figsize=(8, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=T, cmap='coolwarm', s=5)
plt.colorbar(sc, label='Temperature')
plt.title("PCA of Ising Configurations")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.show()
