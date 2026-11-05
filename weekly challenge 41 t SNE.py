import numpy as np
import matplotlib.pyplot as plt

# Week 41: Non-Linear Dimensionality Reduction - t-SNE from Scratch
# Author: Ing. Arturo Javier Borbon Rojas

print("Initializing t-SNE (Probabilistic Dimensionality Reduction) Engine...")
print("-" * 65)

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200, epochs=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = learning_rate
        self.epochs = epochs

    def _pairwise_distances(self, X):
        """Calculates the squared Euclidean distance matrix."""
        sum_X = np.sum(np.square(X), 1)
        return np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    def _get_high_dimensional_affinities(self, distances):
        """Computes Gaussian probabilities (P) in high-dimensional space."""
        # For a full implementation, variance (sigma) is found via binary search using perplexity.
        # Here we use a simplified fixed variance approach for the foundational math.
        sigma = 1.0 
        P = np.exp(-distances / (2 * sigma ** 2))
        np.fill_diagonal(P, 0)
        
        # Normalize to get probabilities
        P = P / np.sum(P, axis=1, keepdims=True)
        # Make the matrix symmetric: (P + P.T) / 2N
        P = (P + P.T) / (2 * P.shape[0])
        return np.maximum(P, 1e-12)

    def fit_transform(self, X):
        n_samples = X.shape[0]
        
        # 1. Compute High-Dimensional Probabilities (Gaussian)
        print("Computing High-Dimensional Gaussian Affinities (P)...")
        distances = self._pairwise_distances(X)
        P = self._get_high_dimensional_affinities(distances)

        # 2. Initialize Low-Dimensional Embeddings randomly
        np.random.seed(42)
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        
        # Momentum variables for Gradient Descent optimization
        Y_velocity = np.zeros_like(Y)
        momentum = 0.8

        print(f"Optimizing Kullback-Leibler Divergence over {self.epochs} epochs:")
        
        for epoch in range(self.epochs):
            # 3. Compute Low-Dimensional Probabilities (Student's t-distribution with 1 degree of freedom)
            dist_Y = self._pairwise_distances(Y)
            # Student-t formula: (1 + ||Yi - Yj||^2)^-1
            Q_unnormalized = 1 / (1 + dist_Y)
            np.fill_diagonal(Q_unnormalized, 0)
            Q = Q_unnormalized / np.sum(Q_unnormalized)
            Q = np.maximum(Q, 1e-12)

            # 4. Calculate Gradient of the KL Divergence
            # dC/dYi = 4 * sum((P_ij - Q_ij) * (Y_i - Y_j) * (1 + ||Y_i - Y_j||^2)^-1)
            gradient = np.zeros_like(Y)
            PQ_diff = P - Q
            
            for i in range(n_samples):
                # Vectorized gradient calculation for efficiency
                grad_i = 4 * np.sum(
                    (PQ_diff[i] * Q_unnormalized[i])[:, np.newaxis] * (Y[i] - Y),
                    axis=0
                )
                gradient[i] = grad_i

            # 5. Gradient Descent Update with Momentum
            Y_velocity = momentum * Y_velocity - self.lr * gradient
            Y += Y_velocity
            
            # Print progress and KL Divergence Loss
            if (epoch + 1) % 200 == 0 or epoch == 0:
                kl_divergence = np.sum(P * np.log(P / Q))
                print(f"   ➔ Epoch {epoch+1:04d} | KL Divergence: {kl_divergence:.4f}")

        return Y

# --- Synthetic Educational Dataset (Non-Linear Structures) ---
# Simulating a complex dataset where linear PCA would fail
# Features: [Grades, Absences, Stress, Tech Access]
np.random.seed(42)
# Cluster 1: High performing, low stress
c1 = np.random.normal(loc=[9, 2, 3, 1], scale=0.5, size=(50, 4))
# Cluster 2: Severe dropout risk
c2 = np.random.normal(loc=[5, 15, 8, 0], scale=1.0, size=(50, 4))
# Cluster 3: The "Invisible" risk (Good grades, but highly stressed & absent)
c3 = np.random.normal(loc=[8, 12, 9, 1], scale=0.8, size=(50, 4))

X_data = np.vstack([c1, c2, c3])
y_labels = np.array([0]*50 + [1]*50 + [2]*50)

# Run custom t-SNE
tsne = TSNE(learning_rate=150, epochs=1000)
Y_2D = tsne.fit_transform(X_data)

print("-" * 65)
print("PROJECTION COMPLETE. Generating visualization...")

# --- Visualization ---
plt.figure(figsize=(8, 6))
colors = ['#2ca02c', '#d62728', '#ff7f0e']
labels = ['Safe', 'High Risk', 'Invisible Risk']

for i in range(3):
    mask = (y_labels == i)
    plt.scatter(Y_2D[mask, 0], Y_2D[mask, 1], 
                c=colors[i], label=labels[i], 
                edgecolor='k', s=80, alpha=0.8)

plt.title("t-SNE: Non-Linear Probabilistic Dimensionality Reduction", fontsize=14, fontweight='bold')
plt.xlabel("Embedded Dimension 1")
plt.ylabel("Embedded Dimension 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()