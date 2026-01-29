# Types of models used in ML and DL
Here's a concise overview of the main model families:

---

**Traditional Machine Learning**

**Linear Models** (Linear Regression, Logistic Regression) — Interpretable, fast, work well when relationships are approximately linear. Used for regression tasks, binary classification, and when explainability matters.

**Tree-Based Models** (Decision Trees, Random Forest, XGBoost, LightGBM) — Handle non-linear relationships, feature interactions, and mixed data types well. Dominant for tabular/structured data like your employee retention project. Used for classification, regression, ranking.

**Support Vector Machines** — Find optimal decision boundaries. Effective for high-dimensional data and smaller datasets. Used for classification and regression.

**Naive Bayes** — Probabilistic classifier assuming feature independence. Fast and works well with limited data. Used for text classification, spam filtering.

**K-Nearest Neighbors** — Instance-based learning using similarity. Simple but computationally expensive at scale. Used for classification, regression, recommendation.

**Clustering Models** (K-Means, DBSCAN, Hierarchical) — Unsupervised grouping of similar data points. Used for customer segmentation, anomaly detection, pattern discovery.

**Dimensionality Reduction** (PCA, t-SNE, UMAP) — Reduce feature space while preserving structure. Used for visualization, preprocessing, noise reduction.

---

**Deep Learning**

**Feedforward Neural Networks (MLPs)** — Layers of connected neurons. Used for tabular data when you have large datasets and complex patterns.

**Convolutional Neural Networks (CNNs)** — Specialized for spatial patterns through learned filters. Dominant for image classification, object detection, medical imaging.

**Recurrent Neural Networks (RNNs, LSTMs, GRUs)** — Process sequential data with memory. Used for time series, speech recognition, language tasks.

**Transformers** — Attention-based architecture handling long-range dependencies. Foundation of modern NLP (BERT, GPT) and increasingly vision tasks. Used for text generation, translation, summarization, question answering.

**Autoencoders** — Learn compressed representations by reconstructing input. Used for anomaly detection, denoising, dimensionality reduction.

**Generative Adversarial Networks (GANs)** — Two networks competing to generate realistic data. Used for image generation, data augmentation, style transfer.

**Diffusion Models** — Generate data by learning to reverse a noise process. Current state-of-art for image generation (Stable Diffusion, DALL-E).

---

**Reinforcement Learning**

**Q-Learning, Deep Q-Networks, Policy Gradient Methods** — Learn optimal actions through environment interaction and rewards. Used for robotics, game playing, recommendation systems, autonomous systems.

---

**Quick Selection Guide**

| Data Type | First Consider |
|-----------|----------------|
| Tabular/structured | Tree-based ensembles |
| Images | CNNs |
| Text | Transformers |
| Sequences/time series | RNNs, Transformers |
| Unlabeled data | Clustering, Autoencoders |
| Decision-making agents | Reinforcement Learning |

For your employee retention work, tree-based models remain the right choice—tabular HR data with moderate size is exactly where they excel.