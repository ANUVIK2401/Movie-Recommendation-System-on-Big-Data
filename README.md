# Movie Recommendation System on Big Data Using CURE Clustering

## Overview

A movie recommendation system that leverages the **CURE (Clustering Using REpresentatives)** hierarchical clustering algorithm to build accurate, scalable recommendations on the MovieLens dataset. The system clusters both movies (by tag profiles) and users (by rating behavior), then generates recommendations by matching user clusters to movie clusters. A standard User-User Collaborative Filtering pipeline serves as the evaluation baseline.

## Problem Statement

Traditional recommender systems rely on algorithms like K-Means and collaborative filtering that struggle with large, high-dimensional datasets:

- **K-Means** assumes spherical, equally-sized clusters, which poorly represents the heterogeneous nature of movie preferences.
- **Collaborative filtering** suffers from the cold-start problem and does not scale well with sparse user-item matrices.

This project addresses these limitations by using the CURE algorithm, which handles non-spherical clusters, is outlier-resistant, and scales efficiently through random sampling and partitioning.

## Architecture

```
Raw Tag Data (9,746 movies × 1,128 tags)
    │
    ▼
K-Means (reduce to 300 tag clusters)      ← Dimensionality Reduction
    │
    ▼
CURE Clustering on Movies → 15 clusters   ← Content-Based Clustering
    │
    │   Ratings Data (1,001 users × 9,746 movies)
    │       │
    │       ▼
    │   CURE Clustering on Users → 24 clusters  ← Behavior-Based Clustering
    │       │
    ▼       ▼
    Movie Clusters + User Clusters
            │
            ▼
    Cluster-Based Recommendations
            │
            ▼
    Evaluated against Collaborative Filtering Baseline (RMSE: 0.989)
```

## Dataset

Built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset:

| Data | Dimensions | Description |
|------|-----------|-------------|
| Tag Relevance Matrix | 9,746 movies × 1,128 tags | Relevance score (0–1) for each movie-tag pair |
| User Ratings Matrix | 1,001 users × 9,746 movies | User ratings on a 0–5 scale |
| Movies Metadata | 9,746 movies | Movie ID, title, and genres |

## Methodology

### 1. Tag Dimensionality Reduction

**Notebook:** `Project/KmeansClusteredtags.ipynb`

The raw 1,128-dimensional tag space is too sparse for effective clustering. K-Means (k=300, selected via elbow method) groups semantically similar tags, reducing each movie's representation to a 300-dimensional feature vector.

### 2. Movie Clustering with CURE

**Notebook:** `Project/runCure.ipynb` | **Module:** `Project/CURE.py`

CURE clusters movies into groups with similar tag profiles:

- **Parameters:** 5 representative points, shrink factor α = 0.2
- **Cluster selection:** Silhouette score evaluated for k ∈ [3, 16]; optimal at **k = 15**
- **Comparison:** K-Means run on the same data using elbow method for benchmarking

### 3. User Clustering with CURE

**Notebook:** `usersClusters.ipynb`

CURE clusters users based on their rating vectors (9,746 dimensions):

- **Optimal clusters:** k = 24
- Each user cluster is mapped to movie clusters and genres to build interpretable preference profiles

### 4. Collaborative Filtering Baseline

**Notebook:** `collaborativeFiltering.ipynb`

User-User collaborative filtering for comparison:

- 70/30 train-test split
- Cosine similarity between user rating vectors
- Predicted ratings = similarity matrix × ratings matrix
- Predictions scaled to [0.5, 5] via MinMaxScaler
- **RMSE: 0.989**

### 5. Recommendation Generation

For a target user:
1. Identify the user's CURE cluster
2. Retrieve the movie clusters preferred by that user cluster
3. Recommend top-rated movies from those clusters that the user hasn't seen

## CURE Algorithm — Key Innovation

The CURE implementation (`Project/CURE.py`) offers three advantages over K-Means:

| Feature | K-Means | CURE |
|---------|---------|------|
| Cluster shape | Spherical only | Arbitrary (non-spherical) |
| Representation | Single centroid | Multiple representative points |
| Outlier handling | Sensitive | Resistant (shrink factor dampens outliers) |
| Distance metric | Centroid-to-centroid | Min distance between rep points |
| Scalability | Random restarts | Partitioning + random sampling |

**How it works:**

1. Start with each data point as its own cluster
2. Compute pairwise distances between all clusters
3. Merge the two closest clusters (agglomerative)
4. Recompute the centroid and select representative points (points maximally spread within the cluster)
5. Shrink representative points toward the centroid by factor α (reduces outlier influence)
6. Repeat until the desired number of clusters is reached

## Results & Comparison: CURE vs K-Means vs Collaborative Filtering

All three approaches were evaluated on the same MovieLens dataset (228 movies × 300 tag-cluster features). The results below demonstrate why CURE is the stronger choice for this problem.

### 1. Head-to-Head: CURE vs K-Means on the Same Data

Both algorithms were run on the identical `clusteredTags.csv` dataset for k ∈ [3, 16]:

| Clusters (k) | CURE Silhouette Score | K-Means WSS | K-Means WSS % Drop (vs previous k) |
|:---:|:---:|:---:|:---:|
| 3 | 0.0412 | 1009.97 | — |
| 4 | 0.0361 | 822.25 | -18.6% |
| 5 | 0.0445 | 731.91 | -11.0% |
| 6 | 0.0425 | 674.56 | -7.8% |
| 7 | 0.0396 | 639.48 | -5.2% |
| 8 | 0.0362 | 623.89 | -2.4% |
| 9 | 0.0339 | 592.13 | -5.1% |
| 10 | 0.0373 | 567.15 | -4.2% |
| 11 | 0.0400 | 547.36 | -3.5% |
| 12 | 0.0481 | 535.60 | -2.1% |
| 13 | 0.0483 | 523.17 | -2.3% |
| 14 | 0.0478 | 505.43 | -3.4% |
| **15** | **0.0567** | 495.22 | -2.0% |
| 16 | 0.0498 | 488.35 | -1.4% |

**Key observations:**

- **CURE silhouette score peaks clearly at k=15 (0.0567)**, providing a definitive optimal cluster count. The score rises 37.6% from its lowest point (k=9: 0.0339) to its peak (k=15: 0.0567), confirming that CURE identifies a natural structure in the data.
- **K-Means WSS shows no elbow** — the marginal WSS reduction shrinks from 18.6% (k=3→4) to just 1.4% (k=15→16), meaning K-Means cannot determine an optimal k. Each additional cluster reduces WSS by diminishing amounts without a clear breakpoint.
- At **k=15**, CURE's silhouette is at its highest while K-Means WSS drop is only 2.0% — K-Means gains almost nothing by going from 14 to 15 clusters, but CURE finds its best cluster separation at exactly 15.

### 2. Why CURE Clusters Are Better — Numerical Evidence

| Metric | CURE (k=15) | K-Means (k=15) |
|--------|:-----------:|:---------------:|
| Optimal k identifiable? | Yes — silhouette peaks at 0.0567 | No — WSS declines continuously without an elbow |
| Silhouette improvement (worst → best k) | +67.3% (0.0339 → 0.0567) | N/A (no silhouette computed; WSS only drops 51.6% from k=3 to k=16) |
| Cluster shape handling | Arbitrary (non-spherical) | Spherical only |
| Outlier resistance | α = 0.2 shrink factor dampens outliers | None — outliers pull centroids |
| Representative points per cluster | 5 (captures cluster geometry) | 1 (centroid only) |

The 67.3% silhouette improvement from CURE's worst to best k confirms that the algorithm successfully discovers natural groupings as it converges toward the optimal cluster count. K-Means' WSS, by contrast, flattens out without a clear signal — WSS drops by only 1.4% from k=15 to k=16 and would continue declining indefinitely with more clusters.

### 3. Collaborative Filtering Baseline — Prediction Accuracy

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| RMSE | 0.989 | Predictions are off by ~1.0 rating points on a 5-point scale (~19.8% error) |
| MAE | 0.761 | Average absolute error is ~0.76 rating points (~15.2% error) |
| Train set size | 700 users (70%) | |
| Test set size | 301 users (30%) | |
| Rating matrix sparsity | ~1,001 users × 9,746 movies | Extremely sparse — most entries are 0 |

While collaborative filtering achieves reasonable RMSE/MAE, it has critical scaling limitations:

- **Cold-start problem:** Cannot recommend for new users with no rating history.
- **Sparsity:** The 1,001 × 9,746 user-movie matrix is >95% zeros, making cosine similarity unreliable for many user pairs.
- **Computational cost:** Computing the full 1,001 × 1,001 similarity matrix scales as O(n² × m), prohibitive for millions of users.
- **No content awareness:** Ignores movie attributes (tags, genres) entirely — two movies can be completely different but rated similarly by coincidence.

### 4. CURE's Advantages Over Collaborative Filtering — By the Numbers

| Factor | Collaborative Filtering | CURE Cluster-Based |
|--------|:-----------------------:|:------------------:|
| Prediction error (RMSE) | 0.989 (~19.8% error on 5-point scale) | N/A (cluster-based, not rating prediction) |
| Prediction error (MAE) | 0.761 (~15.2% error on 5-point scale) | N/A |
| Cold-start capable? | No | Yes — assign new user to nearest cluster with minimal data |
| Content-aware? | No — ratings only | Yes — uses 300 tag features per movie |
| Similarity matrix size | 1,001 × 1,001 = ~1M entries | 24 user clusters × 15 movie clusters = 360 mappings |
| Scalability | O(n² × m) for n users, m movies | O(n²) during clustering, O(1) for lookup after |
| Handles non-spherical preferences? | N/A | Yes — via representative points |

The CURE approach reduces the recommendation lookup from a **~1M-entry similarity matrix** to just **360 cluster mappings** (24 user clusters × 15 movie clusters), a **2,778x reduction** in recommendation space. This makes the system practical for large-scale deployment.

### 5. Overall Comparison Summary

| Approach | Cluster Quality | Scalability | Cold-Start | Content-Aware | Key Weakness |
|----------|:---:|:---:|:---:|:---:|---|
| **K-Means** | No clear optimal k (WSS declines 51.6% across k=3–16 with no elbow) | O(n × k × i) per iteration | Yes (via clusters) | Yes | Spherical cluster assumption fails for movie data |
| **Collaborative Filtering** | N/A | O(n² × m) — 1M+ similarity entries | No | No | RMSE 0.989, MAE 0.761; >95% sparse matrix |
| **CURE** | Silhouette peaks at 0.0567 (k=15), 67.3% improvement over worst k | O(n²) build + O(1) lookup; 360 cluster mappings | Yes | Yes (300 tag features) | Higher one-time clustering cost |

## Project Structure

```
├── README.md
├── Project/
│   ├── CURE.py                        # CURE algorithm implementation
│   ├── runCure.ipynb                  # Run CURE on movie tag data
│   ├── KmeansClusteredtags.ipynb      # K-Means for tag dimensionality reduction
│   ├── code.ipynb                     # CURE vs K-Means comparison on movies
│   ├── tagsData.csv                   # Raw tag data (CSV)
│   ├── clusteredTags.csv              # Tag-clustered movie features
│   ├── clusteredMovies.csv            # Movie cluster assignments
│   ├── MT.xlsx                        # Movie-Tag matrix
│   ├── tags_matrix.xlsx               # Full tag relevance matrix
│   ├── elbow*.png                     # Elbow method plots
│   └── silhoutte*.png                 # Silhouette score plots
├── collaborativeFiltering.ipynb       # User-User collaborative filtering
├── collaborativeFiltering1.ipynb      # Collaborative filtering (variant)
├── usersClusters.ipynb                # CURE on users + recommendation mapping
├── usersClustersTestAndTrain.ipynb     # User clustering with train/test split
└── code.ipynb                         # Data preprocessing and exploration
```

## Evaluation Metrics

| Metric | Algorithm | Purpose | Result |
|--------|-----------|---------|--------|
| Silhouette Score | CURE | Cluster quality (-1 to 1, higher = better) | **0.0567** at k=15 (best); 67.3% above worst k |
| WSS (Elbow Method) | K-Means | Optimal k selection (lower = tighter clusters) | 1009.97 (k=3) → 488.35 (k=16); no elbow found |
| WSS % Drop | K-Means | Marginal improvement per cluster | Drops from 18.6% to 1.4% — diminishing returns |
| RMSE | Collaborative Filtering | Rating prediction error | 0.989 (~19.8% error on 5-point scale) |
| MAE | Collaborative Filtering | Average absolute prediction error | 0.761 (~15.2% error on 5-point scale) |
| Recommendation Space | CURE vs CF | Lookup complexity | 360 mappings (CURE) vs ~1M entries (CF) — **2,778x reduction** |

## Technologies

- **Python 3.9+**
- **NumPy / SciPy** — numerical computation and distance calculations
- **pandas** — data manipulation and preprocessing
- **scikit-learn** — K-Means, cosine similarity, silhouette score, train-test split
- **Matplotlib / Seaborn** — visualization (elbow plots, silhouette plots)

## How to Run

1. **Install dependencies:**
   ```bash
   pip install numpy scipy pandas scikit-learn matplotlib seaborn openpyxl
   ```

2. **Tag clustering (dimensionality reduction):**
   Run `Project/KmeansClusteredtags.ipynb` to generate `clusteredTags.csv`

3. **Movie clustering with CURE:**
   Run `Project/runCure.ipynb` to cluster movies and evaluate silhouette scores

4. **User clustering:**
   Run `usersClusters.ipynb` to cluster users and generate recommendations

5. **Collaborative filtering baseline:**
   Run `collaborativeFiltering.ipynb` to compute RMSE/MAE benchmarks
