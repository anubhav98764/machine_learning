# Practice set: Iris clustering and dimensionality reduction

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load the dataset and select the four numeric feature columns.
df = pd.read_csv("Iris.csv")
feature_columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
X = df[feature_columns]

# Scaling improves K-Means and PCA because the features have different scales.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Apply K-Means and print the cluster centers.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Convert centers back to the original measurement units.
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster centers:")
print(pd.DataFrame(cluster_centers, columns=feature_columns))

# 3. Apply PCA to reduce the dataset to two dimensions.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 2. Visualize the resulting clusters.
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    edgecolor="black",
    alpha=0.8,
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Clusters after PCA")
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Print the explained variance ratio for the two components.
print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)
print("Total variance explained by 2 components:",
      pca.explained_variance_ratio_.sum())