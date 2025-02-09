import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset with more dots
num_points = 300  
X1 = np.random.multivariate_normal([30, 60], [[100, 20], [20, 100]], num_points // 3)
X2 = np.random.multivariate_normal([80, 20], [[100, 20], [20, 100]], num_points // 3)
X3 = np.random.multivariate_normal([100, 80], [[100, 20], [20, 100]], num_points // 3)
data = np.vstack((X1, X2, X3))

# Convert to Pandas DataFrame
df = pd.DataFrame(data, columns=["Feature 1", "Feature 2"])

# Function to initialize centroids randomly
def initialize_centroids(data, k):
    np.random.seed(42)  # Ensure reproducibility
    return data[np.random.choice(data.shape[0], k, replace=False)]

# Function to assign clusters based on Euclidean distance
def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Function to update centroids based on cluster means
def update_centroids(data, labels, k):
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

# K-Means Algorithm Implementation
def kmeans(data, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids

# Run K-Means for k=2 and k=3 and plot results
for k in [2, 3]:
    labels, centroids = kmeans(data, k)

    # Plot clustered data
    plt.figure(figsize=(8, 5))
    for i in range(k):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i}', alpha=0.8, s=20)
    
    # Lighter centroid color with thinner edge
    plt.scatter(centroids[:, 0], centroids[:, 1], c='lightcoral', marker='X', s=150, edgecolors='gray', linewidth=1.5, label='Centroids')
    
    # Customize plot
    plt.title(f"K-Means Clustering (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()