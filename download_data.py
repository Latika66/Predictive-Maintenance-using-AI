import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def main():
    print("Fetching AI4I 2020 Predictive Maintenance Dataset from UCI ML Repository...")
    # fetch dataset 
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
    
    # data (as pandas dataframes) 
    X = ai4i_2020_predictive_maintenance_dataset.data.features 
    y = ai4i_2020_predictive_maintenance_dataset.data.targets 
    
    # combine features and targets
    df = pd.concat([X, y], axis=1)
    
    # save to data folder
    output_path = os.path.join("data", "ai4i2020.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully downloaded and saved to {output_path}")

if __name__ == "__main__":
    main()
