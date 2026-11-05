# weekly-challenge-41-t-SNE
To close out a three-week strategic sprint, I tackled one of the most mathematically beautiful algorithms in Data Science: **t-Distributed Stochastic Neighbor Embedding (t-SNE)**.


While PCA (programmed in Week 34) is excellent for linear data compression, it completely fails to preserve local neighborhood structures when dealing with complex, non-linear relationships. t-SNE solves this by abandoning linear algebra in favor of **Probability and Calculus**. It translates multi-dimensional distances into probabilities and projects them into a 2D map, perfectly untangling complex clusters.

## How it works (The Math)
1. **High-Dimensional Probabilities ($P$):** The algorithm calculates the Euclidean distance between all points in the high-dimensional space and converts them into probabilities using a **Gaussian (Normal) Distribution**. If two points are close, their probability of being "neighbors" is high.
2. **Low-Dimensional Probabilities ($Q$):** It initializes a random 2D map and calculates the probabilities of those same points being neighbors, but this time using a **Student's t-distribution** (which has heavier tails to prevent the "crowding problem" in 2D space).
3. **Kullback-Leibler Divergence:** The algorithm's goal is to make the 2D map ($Q$) look exactly like the original high-dimensional space ($P$). It uses the KL Divergence as the Loss Function:
   $$C = KL(P || Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
4. **Gradient Descent:** It applies the exact derivative of the KL divergence to calculate the gradient, pushing and pulling the points in the 2D plane iteratively until the clusters are perfectly untangled.

## Technical Highlights
* **Pure Calculus Implementation:** The exact gradient matrix calculation for the KL divergence is fully vectorized using pure NumPy for optimal execution speed.
* **Momentum Optimization:** Integrated a physics-based momentum term into the Gradient Descent updater to help the algorithm escape local minima and converge faster.

## Dependencies
* Python 3.14.3
* NumPy
* Matplotlib
