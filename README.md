# **Credit Card Fraud Detection Using XGBoost & Logistic Regression**

## **Project Overview**

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, with the majority of transactions being legitimate. I applied resampling strategies, feature scaling, and trained both Logistic Regression and XGBoost models to compare performance.

To see all the research and experimentation i did to arrive at the best model and optimal results, check the research.ipynb notebook under the notebooks folder while the 
applied and cleaned notebook of the best model and techniques derived from research results awill be found in the main.ipynb notebook.

---

## **Dataset**

* **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Size:** 284,807 transactions, 492 frauds (\~0.172% fraud rate)
* **Features:**

  * `Time`: Seconds since the first transaction
  * `V1`–`V28`: PCA-transformed anonymized features
  * `Amount`: Transaction amount
  * `Class`: Target (0 = legitimate, 1 = fraud)

---

## **Project Goals**

1. Handle extreme class imbalance effectively.
2. Compare the performance of Logistic Regression and XGBoost.
3. Explore the impact of including the `Time` feature.
4. Identify the most important features driving predictions.
5. Build a reproducible and portfolio-ready pipeline.

---

## **Methodology**

### **1. Exploratory Data Analysis (EDA)**

* Checked transaction distributions, fraud frequency by hour, and correlations.
* Observed that fraud tends to spike at certain hours (`Time` feature is informative).

### **2. Data Preprocessing**

* Split features and target into datasets **with Time** and **without Time**.
* Scaled features for Logistic Regression (XGBoost doesn’t require scaling).

### **3. Handling Class Imbalance**

* Tested multiple resampling strategies:

  * None
  * SMOTE (Synthetic Minority Oversampling)
  * Random Undersampling
  * SMOTE + ENN (Combined oversampling + cleaning)

### **4. Model Training**

* **Logistic Regression**: Balanced class weights, scaled features.
* **XGBoost**: Tuned hyperparameters, used `scale_pos_weight` for imbalance.
* Trained separately on each resampled dataset to compare performance.

### **5. Model Evaluation**

* Metrics used:

  * ROC AUC
  * Precision-Recall AUC
  * Confusion Matrix
  * Classification Report
* Observed XGBoost with SMOTE on the dataset **with Time** gave the best results:

  * ROC AUC: 0.9999
  * PR AUC: 1.0
  * F1-score: 0.9999

### **6. Feature Importance**

* Visualized top features contributing to fraud detection using XGBoost.
* Highlights which PCA components and transaction features are most predictive.

---

## **Results**

| Model               | Dataset      | Resampling | ROC AUC | PR AUC | F1-score |
| ------------------- | ------------ | ---------- | ------- | ------ | -------- |
| Logistic Regression | With Time    | SMOTE      | 0.9804  | 0.9977 | 0.9804   |
| XGBoost             | With Time    | SMOTE      | 0.9999  | 1.0    | 0.9999   |
| Logistic Regression | Without Time | SMOTE      | 0.9590  | 0.9930 | 0.9590   |
| XGBoost             | Without Time | SMOTE      | 0.9999  | 1.0    | 0.9999   |

*Note: Best model is XGBoost with Time + SMOTE.*

---

## **How to Run**

1. Clone the repository.
2. Install dependencies:

```bash
conda create -n fraud-detect python=3.11
conda activate fraud-detect
conda install pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

3. Place `creditcard.csv` in the `dataset/` folder.
4. Run the notebook step by step.

---

## **Saved Model**

* XGBoost model saved as `models/model.sav`.
* Can be loaded using `pickle` to make predictions on new transactions.

```python
import pickle
with open("models/model.sav", "rb") as f:
    model = pickle.load(f)
```

---

## **Insights**

* Including `Time` slightly improves fraud detection.
* Resampling with SMOTE yields the best tradeoff between recall and precision.
* XGBoost significantly outperforms Logistic Regression for imbalanced data.
* Feature importance analysis reveals which components of the transaction data are critical.

---

## **Future Work**

* Tune hyperparameters with automated search (GridSearchCV or Optuna).
* Explore ensemble models combining XGBoost and Logistic Regression.
* Deploy as a real-time fraud detection API