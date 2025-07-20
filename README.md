
This project explores user transaction behavior using enhanced wallet-level features derived from raw financial data. The goal is to analyze behavior trends, understand score distributions, and compare rule-based and machine learning-based approaches for wallet scoring.

The core components include:

Preprocessing and EDA on enhanced transaction features.

Credit score visualization and segmentation.

ML vs. Rule-based score comparison via scatter plots.

User segmentation by behavioral metrics.

Contents
File	Description
enhanced_wallet_scores.csv	Main dataset containing user-level financial and behavioral features.
credit_score_distribution.png	Visualization of the distribution of credit scores across users.
ml_vs_rule_score_scatter.png	Scatter plot comparing ML-based and Rule-based wallet scores.
analysis.md	Markdown document summarizing visual insights and key takeaways.
README.md	This file.

📌 Key Features
Full-dataset used for generating insights and comparisons

 Credit score analysis across buckets (e.g., Low, Medium, High)

Wallet feature distributions (transaction count, balance metrics, churn risk)
 Behavioral clustering and segmentation
Evaluation of Machine Learning scoring models against heuristic rules

Visual Insights
1. Credit Score Distribution

A clear right-skewed distribution indicates a larger user base with lower scores, highlighting inclusion needs or tighter risk controls.

2. ML vs Rule-based Score

The scatter plot shows linear correlation with minor outliers, suggesting ML scores largely agree with rule-based logic while capturing some edge cases better.

 Sample Features Used
txn_count — Total number of transactions

wallet_balance — Running balance over 6 months

avg_debit_credit_diff — Average net spend behavior

last_active_days — Days since last transaction (proxy for churn)

risk_bucket — Rule-assigned risk segment

ml_score — ML-predicted wallet score (0–100)

 How to Use
Clone the repository

bash
Copy
Edit
git clone https://github.com/Ashwinxxx/Transaction_behavior.git
cd Transaction_behavior
View analysis
Open analysis.md to read insights and conclusions.


