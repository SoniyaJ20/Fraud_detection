import pandas as pd
import numpy as np

def generate_mock_data(n_samples=1000):
    np.random.seed(42)
    normal = np.random.normal(loc=50, scale=10, size=(int(n_samples*0.95), 2))
    fraud = np.random.normal(loc=120, scale=20, size=(int(n_samples*0.05), 2))
    
    data = np.vstack([normal, fraud])
    df = pd.DataFrame(data, columns=['amount', 'distance_from_home'])
    df['label'] = [0]*len(normal) + [1]*len(fraud)
    
    # Save as creditcard.csv so your other scripts can find it
    df.to_csv('creditcard.csv', index=False)
    print("Dataset created: creditcard.csv")

if __name__ == "__main__":
    generate_mock_data()