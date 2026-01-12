import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def train_main_model():
    print("Loading dataset...")
    df = pd.read_csv('creditcard.csv') 

    # Check if this is the real Kaggle file or your mock file
    if 'Class' in df.columns:
        X = df.drop(['Class', 'Time'], axis=1)
        contamination_rate = 0.0017
    else:
        # Fallback for mock data
        X = df[['amount', 'distance_from_home']]
        contamination_rate = 0.05

    scaler = StandardScaler()
    # Apply scaling to the 'amount' column
    X_scaled = X.copy()
    if 'amount' in X.columns:
        X_scaled['amount'] = scaler.fit_transform(X[['amount']])
    
    print("Training Aegis...")
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    model.fit(X_scaled)

    joblib.dump(model, 'aegis_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Success! Model and Scaler saved.")

if __name__ == "__main__":
    train_main_model()