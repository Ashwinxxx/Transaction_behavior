DeFi Wallet Credit Scoring – Aave V2 Protocol (100K+ Transactions)

 Problem Statement

This project aims to develop a machine learning modelthat assigns a **credit score (0–1000)** to each wallet address based on its historical transaction behavior within the Aave V2 protocol. The score reflects responsible or risky behavior, with **higher scores** indicating **trustworthy wallets** and **lower scores** indicating **potentially exploitative or bot-like wallets.
 Input

- Format: Raw JSON file containing transaction-level data per wallet
- Actions included:  
  - `deposit`
  - `borrow`
  - `repay`
  - `redeemunderlying`
  - `liquidationcall`

---

## 🏗️ Approach & Architecture

### ✅ **1. Data Parsing**
- Efficiently read large JSON file (~87MB) using `ujson` and `pandas`.
- Flattened nested transaction histories per wallet.

### ✅ **2. Feature Engineering**
Key wallet-level features engineered:
| Category               | Feature Examples                                                 |
|------------------------|------------------------------------------------------------------|
| **Volume**             | Total deposits, borrows, repayments, redemptions                |
| **Frequency**          | Transaction count per type, total unique days active            |
| **Temporal Behavior**  | Mean/median time between transactions                           |
| **Risk Signals**       | Ratio of `liquidationcall` to borrows, early repays, etc.       |
| **Diversity**          | Number of unique asset types interacted with                    |
| **Efficiency**         | Ratio of repay-to-borrow volume, usage of full repayment, etc.  |

All features were normalized and standardized. Outlier detection applied via IQR.

### ✅ **3. Scoring Model**
- **Algorithm**: `XGBoostRegressor` (chosen for interpretability, performance, and robustness)
- **Target Score**: A normalized credit score ∈ `[0, 1000]`
- **Score Mapping**:
  - Top ~10%: Highly responsible
  - Middle ~60%: Normal users
  - Bottom ~10–20%: Potential exploiters or bots

### ✅ **4. Script Execution**
Single script:  
```bash
python score_wallets.py --input user-transactions.json --output scores.csv
