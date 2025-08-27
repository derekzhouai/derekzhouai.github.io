---
title:  "Understanding K-Means Clustering in Machine Learning"
date:   2025-08-27 10:00:00 +0800
categories: [unsupervised learning, clustering]
tags: [k-means]
math: true
---

Clustering is one of most important techniques in unsupervised learning. Among various clustering algorithms, **K-Means** stands out as one of the simplest and most widely used method. It partitions data into groups (called clusters) such that points in the same cluster are more similar to each other than to those in other clusters.

In this post, we'll walk through **intuition, algorithm steps, mathematical formulation**, and even a **Python implementation** of K-Means.

## Intuition Behind K-Means

The goal of K-Means is to find **k clusters (centroids)** and assign each data point to the closest center (centroid). By doing so, we minimize the overall variance within clusters.

Think of it as repeatedly shuffling data points between different groups until the groups are as "tight" as possible.

## The Algorithm (Step by Step)

1. **Choose the number of clusters, k**<br>
This is a hyperparameter you decide before running the algorithm.
2. **Initialize cluster centroids**<br>
Randomly select k points from the dataset as initial centroids.
3. **Assignment step**<br>
For each point, compute the distance to all centroids and assign it to the closest one.
4. **Update step**<br>
For each cluster, recalculate its centroid as the mean of all points assigned to it.
5. **Repeat**<br>
Alternate between assignment and update steps until the centroids no longer change significantly, or until a maximum number of iterations is reached.

## Mathematical Objective

The objective of K-Means is to minimize the **sum of squared distances** between data points and their assigned cluster centroids:

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2
$$

where:
- $J$ is the objective function (cost function)
- $k$ is the number of clusters
- $C_i$ is the set of points in cluster $i$
- $\mu_i$ is the centroid of cluster $i$
- $$\| x - \mu_i \|^2$$ is the squared Euclidean distance between point $x$ and centroid $\mu_i$

In other words, we want each cluster to be as compact as possible.

## Pseudocode

Here's a concise pseudocode for K-Means:

```
Input: dataset X, number of clusters k
Output: cluster centroids μ, cluster assignments

1. Randomly initialize k cluster centroids μ
2. Repeat until convergence:
   a. Assign each data point to the closest cluster centroid μ
   b. Update cluster centroids μ by computing the mean of all points in each cluster
```

## Python Implementation (From Scratch)

You can find the full code from [here](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/understanding_kmeans.ipynb).

```python
import numpy as np

def kmeans(X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform K-Means clustering.
    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
    Returns:
        tuple[np.ndarray, np.ndarray]: Final cluster centers and labels.
    """
    # Randomly initialize cluster centers
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Compute distances from points to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Recalculate centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids

    return centroids, labels
```

## Example Visualization

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Run KMeans
centroids, labels = kmeans(X, k=4)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
plt.show()
```

This will produce a scatter plot with data points colored by cluster, and red X markers representing centroids.

![](/assets/img/posts/kmeans_cluster.png){: .normal }


## Practical Considerations

- **Choosing k:** Methods like _Elbow Method_ and _Silhouette Score_ can help determine a good number of clusters.
- **Initialization sensitivity:** Random initialization can lead to different results. A common improvement is **K-Means++**, which chooses better starting centroids.
- **Cluster shape limitations:** K-Means works best for spherical, equally sized clusters. It struggles with non-convex shapes or clusters of very different densities.

## Conclusion

K-Means clustering is a fundamental and powerful technique for partitioning data. Its simplicity makes it a great first choice for unsupervised learning tasks, though it has limitations. By understanding how it works under the hood - and even implementing it from scratch - you can gain a much deeper intuition that goes beyond just using it as a black box.
