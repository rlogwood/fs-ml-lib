
# EDA - Exploratory Data Analysis
EDA in AI/ML is the early, systematic exploration of a dataset to understand its structure, quality, and relationships before modeling. It typically includes:
  - Data overview: size, data types, feature descriptions, target distribution
  - Data quality: missing values, duplicates, outliers, inconsistent labels
   - Univariate analysis: distributions, summary stats, skewness/kurtosis
   - Bivariate/multivariate analysis: correlations, group comparisons, feature interactions
   - Feature behavior vs. target: class imbalance, signal strength, leakage checks
   - Visualization: histograms, box plots, scatter plots, heatmaps, pair plots
   - Preliminary preprocessing ideas: scaling, encoding, transformations, feature selection
   - Hypotheses and questions for modeling: what patterns look predictive, what’s noisy

In short: EDA is the “data understanding and sanity‑check” phase that guides modeling choices and reduces surprises later.


# Steps After EDA in ML and AI workflows
After loading, inspecting, and EDA, the typical ML workflow continues with these categories:

**Data Preprocessing** — Handling missing values, removing duplicates, encoding categorical variables, and scaling/normalizing features (you've already worked through this with one-hot encoding and StandardScaler).

**Feature Engineering** — Creating new features, selecting important features, and transforming existing ones to improve model performance.

**Data Splitting** — Dividing data into training, validation, and test sets to enable unbiased model evaluation.

**Model Selection & Training** — Choosing appropriate algorithms and fitting them to the training data (you've explored Logistic Regression, Random Forest, and XGBoost here).

**Model Evaluation** — Assessing performance using metrics appropriate to the problem—in your case, focusing on recall and confusion matrices given the cost of missed departures.

**Hyperparameter Tuning** — Optimizing model parameters through techniques like grid search or random search to improve performance.

**Model Interpretation** — Understanding what the model learned and why it makes certain predictions (SHAP falls here).

**Deployment & Monitoring** — Putting the model into production and tracking its performance over time (your Streamlit/Gradio interface work fits this stage).

These aren't always strictly sequential—you'll often iterate back to earlier stages as you learn more about what works for your specific problem.


# Where does dropping highly correlated features fit in?
Dropping highly correlated features fits into **Feature Engineering**, specifically the **Feature Selection** subset of that category.

The reasoning: you're making decisions about which features to retain based on their relationships with other features, not cleaning data quality issues. Highly correlated features provide redundant information, so removing one of a correlated pair reduces dimensionality without losing predictive signal.

That said, it's worth noting a few things for your retention project:

**When it matters most** — Multicollinearity is primarily a concern for linear models like Logistic Regression, where it can inflate coefficient variance and make interpretation unreliable. Tree-based models (Random Forest, XGBoost) handle correlated features naturally since they can simply choose one over the other at split points.

**Trade-off to consider** — Since you're prioritizing predictive performance (catching departures) over coefficient interpretation, and you're using ensemble methods, dropping correlated features may have minimal impact on your results. It's often more beneficial for model simplicity and training speed than for accuracy.

**Typical approach** — Calculate a correlation matrix, identify pairs above a threshold (commonly 0.7–0.9), then decide which of each pair to drop based on domain knowledge or individual feature importance.

Would you like to explore correlation analysis on your employee dataset?


# Clarification of Data Instpection, EDA and Feature Engineering
Reviewing feature distributions fits primarily into **EDA**.

The purpose determines the category:

**Data Inspection** — "What do I have?" Checking data types, shape, missing values, basic structure. You might glance at distributions here, but it's superficial.

**EDA** — "What does my data look like and why?" This is where distribution analysis lives. You're examining skewness, outliers, multimodality, range, and spread to understand patterns and inform decisions.

**Feature Engineering** — "How should I transform this?" You act on what you learned from EDA—applying log transforms to skewed features, binning, or capping outliers.

---

In practice, they blur together. When you used histplot with hue to examine satisfaction levels by departure status in your retention project, that was EDA—you were exploring how distributions differ between groups. When you decided to cap extreme working hours rather than remove them (because those outliers were predictive signals), that decision bridged EDA insight into a Feature Engineering action.

A simple way to think about it: **observing** distributions is EDA, **transforming** them is Feature Engineering.