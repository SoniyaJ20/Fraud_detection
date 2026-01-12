import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# 1. Load Data
df = pd.read_csv('creditcard.csv')
X = df[['amount', 'distance_from_home']]
y_true = df['label']

# 2. Train Isolation Forest
# contamination is the expected % of fraud
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# 3. Predict
# Isolation Forest returns -1 for outliers and 1 for inliers
y_pred = model.predict(X)
y_pred_converted = [1 if x == -1 else 0 for x in y_pred]

# 4. Evaluate (Session 3 focus: Precision/Recall)
print("Baseline Model Performance:")
print(classification_report(y_true, y_pred_converted))