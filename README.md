# Mapping the MovieLens Ecosystem: A Scalable Multi-Stage Recommender Framework

This repository contains a complete, end-to-end implementation of a large-scale recommendation system. The project systematically progresses through five stages of development: exploratory data analysis (EDA), baseline bias modeling, Neural Collaborative Filtering (NCF), personalized ranking, and latent semantic visualization.

---

## Project Overview

The objective of this project is to architect a scalable system capable of predicting user preferences and ranking items within a highly sparse interaction matrix. The framework transitions from traditional Matrix Factorization via **Alternating Least Squares (ALS)** to non-linear **Deep Learning** architectures.

### Key Technical Achievements

- **Scale**: Optimized to handle 20,000,263 ratings across 200,948 users and 84,432 movies.
- **Performance**: Achieved a final Test RMSE of **~0.875** using Latent Factor Models.
- **Innovation**: Captured non-linear user-item interactions through Deep Learning and visualized embedding spaces.

---

## Phase 1: Exploratory Data Analysis (EDA)

Before modeling, we characterize the dataset properties to identify critical system constraints.

### 1. Power-Law Distributions

Both user activity and movie popularity follow pronounced scale-free characteristics. The probability $P(x)$ that a movie has $x$ ratings follows the Power-Law formula:

$$
P(x) \propto x^{-\alpha}
$$

Visualized through Log-Log Probability Density Functions (PDF), the linear relationship confirms that a small minority of "blockbuster" movies and "power users" dominate the interaction matrix:

$$
\log(P(x)) = -\alpha \log(x) + C
$$

### 2. Rating Density & Bias

The dataset density is extremely low ($\approx 0.118\%$). The dataset shows a distinct positive bias, with the most prevalent ratings occurring between **3.0 and 4.0**.

---

## Phase 2: Matrix Factorization (MF-ALS)

We model user preferences by decomposing the sparse rating matrix into low-rank latent factors.

### Mathematical Formulation

The predicted rating for user $u$ and item $i$ is calculated by combining global, user, and item effects with latent interactions:

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u
$$

Where:

- $\mu$ is the global mean,
- $b_u$ and $b_i$ are user and item biases,
- $q_i^T p_u$ is the dot product of item and user latent vectors.

### Optimization

We minimize the regularized squared error (loss function) to prevent overfitting:

$$
\min_{b, p, q} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2 + \lambda (\|q_i\|^2 + \|p_u\|^2 + b_u^2 + b_i^2)
$$

Using **Alternating Least Squares (ALS)**, we solve for the optimal user vector $p_u$ analytically by fixing all item vectors $q_i$:

$$
p_u = \left( \sum_{i \in \mathcal{I}_u} q_i q_i^T + \lambda I \right)^{-1} \sum_{i \in \mathcal{I}_u} (r_{ui} - \mu - b_u - b_i) q_i
$$

Our implementation stabilized at:

- **Training RMSE**: ~0.835  
- **Test RMSE**: ~0.875

Root Mean Squared Error is defined as:

$$
RMSE = \sqrt{\frac{1}{|\mathcal{K}|} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2}
$$

---

## Phase 3: Neural Collaborative Filtering (NCF)

To overcome the limitations of linear dot products, we implemented a Deep Learning architecture. The linear interaction is replaced by a Multi-Layer Perceptron (MLP) function $f$ with parameters $\Theta$:

$$
\hat{r}_{ui} = f_{MLP}(p_u, q_i \mid \Theta)
$$

---

## Phase 4: Personalized Ranking & Top-N

Moving beyond pointwise error (RMSE), we optimized for **Ranking** using the Bayesian Personalized Ranking (BPR) logic to maximize the probability that a user prefers an observed item $i$ over an unobserved item $j$:

$$
\mathcal{L}_{BPR} = \sum_{(u,i,j) \in \mathcal{D}} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) - \lambda_\Theta \|\Theta\|^2
$$

We evaluate these rankings using **Normalized Discounted Cumulative Gain (NDCG)**:

$$
NDCG_k = \frac{DCG_k}{IDCG_k}, \quad DCG_k = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

### Case Study: Recommendations for a LOTR Fan

| Rank | Movie ID | Predicted Score |
|------|----------|-----------------|
| 1    | 75744    | 2.2319          |
| 2    | 72996    | 2.1288          |
| 3    | 39387    | 1.9538          |

---

## Phase 5: Latent Semantic Visualization

The final stage validates the model's semantic "understanding." We calculate the semantic center (centroid) of a genre $G$ by averaging latent vectors:

$$
\theta_G = \frac{1}{|G|} \sum_{i \in G} q_i
$$

By projecting these high-dimensional genre embeddings into 2D space, we observe clear clustering of semantically similar categories (e.g., Sci-Fi and Fantasy).

---

## Repository Structure

- **Practical 1**: Data exploration, power laws, and sparsity analysis.
- **Practical 2**: Matrix Factorization implementation and RMSE tracking.
- **Practical 3**: Neural Collaborative Filtering and deep learning models.
- **Practical 4**: Ranking models and personalized top-N generation.
- **Practical 5**: Visualization of the latent semantic space and genre embeddings.

---

## Datasets Used

- **MovieLens 20M Dataset**: [Download here](https://grouplens.org/datasets/movielens/20m/)
- **MovieLens Latest Small Dataset**: [Download here](https://grouplens.org/datasets/movielens/latest/)

---

## Usage

1. **Clone the repo**

```bash
git clone https://github.com/Abalo39/Machine_learning_at_scale.git
