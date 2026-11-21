# Mini Project 3: Iris Flower Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


iris = pd.read_csv("Iris.csv")
feature_columns = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
X = iris[feature_columns]
true_labels = iris["Species"]

# Scaling makes K-Means treat all four measurements consistently.
X_scaled = StandardScaler().fit_transform(X)

# Apply K-Means with k = 3.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
predicted_clusters = kmeans.fit_predict(X_scaled)

# Match arbitrary K-Means cluster numbers to species names.
# Linear assignment maximizes agreement in the contingency table.
species_names = sorted(true_labels.unique())
species_to_number = {
    species: number for number, species in enumerate(species_names)
}
true_numbers = true_labels.map(species_to_number).to_numpy()

contingency = pd.crosstab(predicted_clusters, true_numbers).to_numpy()
row_indices, column_indices = linear_sum_assignment(contingency.max() - contingency)
cluster_to_species = {
    row: species_names[column]
    for row, column in zip(row_indices, column_indices)
}
predicted_labels = pd.Series(
    [cluster_to_species[cluster] for cluster in predicted_clusters],
    name="Predicted Species",
)

# Compare the assigned clusters with the true labels.
comparison = pd.crosstab(
    true_labels,
    predicted_labels,
)
print("Predicted vs True Labels:")
print(comparison)
print("\nCluster-aligned accuracy:",
      accuracy_score(true_labels, predicted_labels))
print("Adjusted Rand Score:",
      adjusted_rand_score(true_labels, predicted_clusters))

# Visualize clusters using petal length and petal width.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

scatter = axes[0].scatter(
    iris["PetalLengthCm"],
    iris["PetalWidthCm"],
    c=predicted_clusters,
    cmap="viridis",
    edgecolor="black",
)
axes[0].set_title("K-Means Predicted Clusters")
axes[0].set_xlabel("Petal Length (cm)")
axes[0].set_ylabel("Petal Width (cm)")
plt.colorbar(scatter, ax=axes[0], label="Cluster")

species_colors = {species: index for index, species in enumerate(species_names)}
true_colors = [species_colors[species] for species in true_labels]
scatter = axes[1].scatter(
    iris["PetalLengthCm"],
    iris["PetalWidthCm"],
    c=true_colors,
    cmap="viridis",
    edgecolor="black",
)
axes[1].set_title("True Iris Species")
axes[1].set_xlabel("Petal Length (cm)")
axes[1].set_ylabel("Petal Width (cm)")
plt.colorbar(
    scatter,
    ax=axes[1],
    ticks=range(len(species_names)),
    label="Species",
).ax.set_yticklabels(species_names)

plt.tight_layout()
plt.show()