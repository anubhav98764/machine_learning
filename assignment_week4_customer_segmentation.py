# Option 3: Customer Segmentation using Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


customers = pd.read_csv("Mall_Customers.csv")

customer_id = "CustomerID"
age_column = "Age"
income_column = "Annual Income (k$)"
spending_column = "Spending Score (1-100)"

cluster_features = [
    age_column,
    income_column,
    spending_column,
]
X = customers[cluster_features]

# Scaling gives Age, Income, and Spending Score comparable influence.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
customers["Cluster"] = kmeans.fit_predict(X_scaled)

# Rank clusters by their average spending score, then assign business labels.
cluster_spending = (
    customers.groupby("Cluster")[spending_column]
    .mean()
    .sort_values()
)
cluster_order = cluster_spending.index.tolist()
spending_labels = {
    cluster: (
        "Low Spenders"
        if rank < 2
        else "Medium Spenders"
        if rank < 4
        else "High Spenders"
    )
    for rank, cluster in enumerate(cluster_order)
}
customers["Customer Group"] = customers["Cluster"].map(spending_labels)

print("Cluster Summary:")
print(
    customers.groupby("Customer Group")[cluster_features]
    .mean()
    .round(2)
)

# Visualize clusters by annual income and spending score.
plt.figure(figsize=(9, 6))
for group in customers["Customer Group"].unique():
    group_data = customers[customers["Customer Group"] == group]
    plt.scatter(
        group_data[income_column],
        group_data[spending_column],
        s=60,
        alpha=0.8,
        label=group,
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Additional business view: Age versus Spending Score.
plt.figure(figsize=(9, 6))
for group in customers["Customer Group"].unique():
    group_data = customers[customers["Customer Group"] == group]
    plt.scatter(
        group_data[age_column],
        group_data[spending_column],
        s=60,
        alpha=0.8,
        label=group,
    )

plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Spending Segments by Age")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
