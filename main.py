import pandas as pd
import matplotlib.pyplot as plt
import numpy as np #
scores_file_path = 'enhanced_wallet_scores.csv'

try:
    # Load the scores DataFrame from the CSV file
    scores = pd.read_csv(scores_file_path, index_col='wallet')
    print(f"Successfully loaded scores from '{scores_file_path}'.")

    if not scores.empty:
        # Create a histogram of the enhanced_credit_score
        plt.figure(figsize=(10, 6))
        plt.hist(scores['enhanced_credit_score'], bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Enhanced DeFi Credit Scores')
        plt.xlabel('Credit Score (0-1000)')
        plt.ylabel('Number of Wallets')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig('credit_score_distribution.png')
        print("Generated 'credit_score_distribution.png' showing the distribution of scores.")
        if 'ml_score' in scores.columns and 'rule_score' in scores.columns:
          plt.figure(figsize=(10, 6))
          plt.scatter(scores['ml_score'], scores['rule_score'], alpha=0.6)
          plt.title('ML Score vs Rule-Based Score')
          plt.xlabel('ML Score')
          plt.ylabel('Rule-Based Score')
          plt.grid(True, linestyle='--', alpha=0.6)
          plt.tight_layout()
          plt.savefig('ml_vs_rule_score_scatter.png')
          print("Generated 'ml_vs_rule_score_scatter.png'.")

    else:
        print("The loaded scores DataFrame is empty. No plots generated.")

except FileNotFoundError:
    print(f"Error: The file '{scores_file_path}' was not found. Please ensure 'main.py' has been run successfully to generate this file.")
except Exception as e:
    print(f"An error occurred while loading or plotting scores: {e}")