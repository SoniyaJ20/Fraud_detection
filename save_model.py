import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load data and train (Simplified for this step)
df = pd.read_csv('creditcard.csv')
X = df[['amount', 'distance_from_home']]
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# 2. Save the model to a file
joblib.dump(model, 'fraud_model.joblib')
print("Model saved as fraud_model.joblib")