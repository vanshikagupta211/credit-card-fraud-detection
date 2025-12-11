📌 Credit Card Fraud Detection — Machine Learning Project
By Vanshika Gupta
📖 Project Overview

This project aims to detect fraudulent credit card transactions using machine learning.
The dataset is highly imbalanced (only ~0.17% fraud), which makes fraud detection challenging.

We handle:

Severe class imbalance

Skewed transaction amounts

PCA-transformed features (V1–V28)

False-negative reduction (priority in fraud systems)

🧠 Business Problem

Fraudulent transactions cause major financial losses.
The main goal is to detect fraud early, focusing on:

✔ High Recall → catching most fraud cases
✔ Low False Negatives → avoid missing fraud
✔ Interpretability → features & importance

📊 EDA Highlights

Fraud cases are extremely rare → only 0.17%

Amount values are highly skewed → log-transform improves modeling

Fraud shows certain time-based patterns

PCA features cannot use VIF (no multicollinearity issue)

🔧 ML Pipeline

Load and explore dataset

Create Amount_log (log-transformed amount)

Train-test split (stratified)

Scale features (train-only)

Apply SMOTE (train-only)

Train Logistic Regression & Random Forest

Evaluate using:

Confusion Matrix

ROC-AUC

Precision-Recall AUC

Compare models

Show predictions on unseen samples

🔥 Model Performance Comparison
Model	Precision	Recall	F1	ROC-AUC
Logistic Regression	Low	High	Low	~0.96
Random Forest (Final Model)	High	High	Best	0.97+

👉 Random Forest is selected as the final model.

📈 ROC Curve

(Insert your ROC image here — drag and upload to GitHub)

⚠ Business Insights

Fraud is highly imbalanced, requiring special handling

Log-transformed Amount gives clearer fraud patterns

PCA features still carry strong fraud signals

Random Forest reduces the most false negatives

The model is suitable for real-time fraud scoring

📝 Conclusion

SMOTE, scaling, and preprocessing were applied only on training

No data leakage occurred

Random Forest model gives best trade-off between recall & precision

Model is deployment-ready

🧪 Sample Prediction
Prediction Table (with fraud probability)

📦 Installation
pip install -r requirements.txt

▶ How to Run
Open the Jupyter notebook:
jupyter notebook

Run:
Credit_Card_Fraud_Detection.ipynb

🙋‍♀️ Author

Vanshika Gupta
Data Analyst | Data Science | Machine Learning