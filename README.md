# Mapping the MovieLens Ecosystem: A Scalable Multi-Stage Recommender Framework

This repository contains a complete, end-to-end implementation of a large-scale recommendation system using the **MovieLens 20M dataset**. The project systematically progresses through five stages of development: exploratory data analysis (EDA), baseline bias modeling, Neural Collaborative Filtering (NCF), personalized ranking, and latent semantic visualization.

## 🚀 Project Overview
The objective of this project is to architect a scalable system capable of predicting user preferences and ranking items within a highly sparse interaction matrix. The framework transitions from traditional Matrix Factorization via **Alternating Least Squares (ALS)** to non-linear **Deep Learning** architectures.

### Key Technical Achievements
* [cite_start]**Scale**: Optimized to handle over 20 million ratings across 200,948 users and 84,432 movies[cite: 13, 14, 21].
* [cite_start]**Performance**: Achieved a final Test RMSE of **~0.875** using Latent Factor Models[cite: 80, 83].
* [cite_start]**Innovation**: Captured non-linear user-item interactions through Deep Learning[cite: 82, 83].

---

## 📊 Phase 1: Exploratory Data Analysis (EDA)
Before modeling, we characterize the dataset properties to identify critical system constraints.

### 1. Power-Law Distributions
[cite_start]Both user activity and movie popularity follow pronounced scale-free characteristics[cite: 91, 100]. [cite_start]Visualized through Log-Log Probability Density Functions (PDF), the linear relationship confirms that a small minority of "blockbuster" movies and "power users" dominate the interaction matrix[cite: 97, 107].



### 2. Rating Density & Bias
[cite_start]The dataset shows a distinct positive bias, with the most prevalent ratings occurring between **3.0 and 4.0**[cite: 2, 4, 34]. The system must account for this global shift to avoid skewed predictions.

---

## 🧮 Phase 2: Matrix Factorization (MF-ALS)
We model user preferences by decomposing the sparse rating matrix into low-rank latent factors.

### Mathematical Formulation
The predicted rating for user $u$ and item $i$ is calculated as:
$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$$

### Optimization
We minimize the regularized squared error using **Alternating Least Squares (ALS)**. [cite_start]Our implementation achieved rapid convergence within **15 iterations**, stabilizing at a training RMSE of ~0.835 and a test RMSE of ~0.875[cite: 83, 87, 88].



---

## 🧠 Phase 3: Neural Collaborative Filtering (NCF)
[cite_start]To overcome the limitations of linear dot products, we implemented a Deep Learning architecture[cite: 82, 83]. This model utilizes embedding layers followed by multi-layered perceptrons (MLP) to learn complex, non-linear relationships between users and items.

---

## 🎯 Phase 4: Personalized Ranking & Top-N
Moving beyond pointwise error (RMSE), we optimized the system for **Ranking**. The model scores candidate movies to generate a personalized Top-10 list for specific user personas.

### Case Study: Recommendations for a LOTR Fan
The model successfully identified high-relevance items for a user interested in the *Lord of the Rings* franchise:

| Rank | Movie ID | Predicted Score |
| :--- | :--- | :--- |
| 1 | 75744 | 2.2319 |
| 2 | 72996 | 2.1288 |
| 3 | 39387 | 1.9538 |

---

## 🎨 Phase 5: Latent Semantic Visualization
The final stage validates the model's semantic "understanding" of the catalog. By projecting high-dimensional genre embeddings into 2D space, we observe clear clustering of semantically similar categories (e.g., Sci-Fi and Fantasy).



---

## 🛠️ Repository Structure
* [cite_start]**Practical 1**: Data exploration, power laws, and sparsity analysis[cite: 91, 100].
* [cite_start]**Practical 2**: Matrix Factorization implementation and RMSE tracking[cite: 82, 88].
* [cite_start]**Practical 3**: Neural Collaborative Filtering and deep learning models[cite: 82, 83].
* **Practical 4**: Ranking models and personalized top-N generation.
* **Practical 5**: Visualization of the latent semantic space and genre embeddings.

---

## 🔧 Usage
1. **Clone the repo**: `git clone https://github.com/Abalo39/Machine_learning_at_scale.git`
2. **Download Data**: Use the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/).
3. **Run**: Open the `.ipynb` files in Jupyter or Google Colab to reproduce the figures.
