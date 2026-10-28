"""Educational K-Means customer-segmentation demo using synthetic data.

The generated data is random and is not evidence of real customer behaviour.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
N_CUSTOMERS = 250
N_CLUSTERS = 5

def main():
    # Create reproducible synthetic data. Values are arbitrary demo units,
    # not real customer records or a validated business dataset.
    rng = np.random.default_rng(RANDOM_SEED)
    customers = pd.DataFrame({
        "AnnualIncome": rng.integers(20, 110, size=N_CUSTOMERS),
        "SpendingScore": rng.integers(5, 95, size=N_CUSTOMERS),
    })

    feature_names = customers.columns.tolist()
    features = customers[feature_names]

    # Scale features before K-Means so each feature contributes on a
    # comparable scale to the distance calculation.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_SEED,
    )
    customers["Cluster"] = model.fit_predict(scaled_features)

    score = silhouette_score(scaled_features, customers["Cluster"])
    print("K-Means clustering demo (synthetic data)")
    print(f"Customers generated: {len(customers)}")
    print(f"Clusters: {N_CLUSTERS}")
    print(f"Silhouette score: {score:.3f}")
    print("\nCluster profiles (means are in the original demo units):")
    profile = (
        customers.groupby("Cluster")[feature_names]
        .mean()
        .round(1)
    )
    profile["CustomerCount"] = customers.groupby("Cluster").size()
    print(profile.to_string())

    # Demonstrate assignment of one hypothetical customer.
    example_customer = pd.DataFrame(
        [{"AnnualIncome": 65, "SpendingScore": 40}],
        columns=feature_names,
    )
    example_scaled = scaler.transform(example_customer)
    predicted_cluster = int(model.predict(example_scaled)[0])
    print(
        "\nExample only: income=65, spending score=40 "
        f"was assigned to cluster {predicted_cluster}."
    )
    print("Cluster numbers are arbitrary labels, not customer-value rankings.")

    # Convert scaled centroids back to the original feature scale for plotting.
    centers = scaler.inverse_transform(model.cluster_centers_)
    labels = customers["Cluster"].to_numpy()

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        customers["AnnualIncome"],
        customers["SpendingScore"],
        c=labels,
        cmap="viridis",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.3,
    )
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        c="black",
        label="Cluster centroids",
    )
    plt.xlabel("Annual income (synthetic demo units)")
    plt.ylabel("Spending score (synthetic demo units)")
    plt.title("K-Means clusters on randomly generated data")
    plt.legend()
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
