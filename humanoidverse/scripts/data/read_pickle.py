import pickle
import numpy as np
import os
import sys
import pandas as pd
import joblib

# Set numpy print options
np.set_printoptions(threshold=100, precision=3)

def print_dict_info(d, prefix=""):
    """Print information about a dictionary and its contents."""
    for key, value in d.items():
        print(f"{prefix}Key: {key}, Type: {type(value)}")
        
        if isinstance(value, dict):
            print_dict_info(value, prefix + "  ")
        elif isinstance(value, np.ndarray):
            print(f"{prefix}  Shape: {value.shape}")
            print(f"{prefix}  Sample: {value[:2] if value.size > 2 else value}")
        else:
            print(f"{prefix}  Value: {value}")

def main():
    file_path = "humanoidverse/data/motions/adam/singles/0-BMLmovi_Subject_58_F_MoSh_Subject_58_F_12_poses.pkl"
    print(f"Loading pickle file: {file_path}")
    
    try:
        methods = [
            ('standard pickle', lambda f: pickle.load(open(f, 'rb'))),
            ('pandas', lambda f: pd.read_pickle(f)),
            ('joblib', lambda f: joblib.load(f))
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"Trying {method_name}...")
                data = method_func(file_path)

                np.savetxt(f"dof_{file_path.split('/')[-1][:-4]}.txt", data[file_path.split("/")[-1][:-4]]["dof"])
                
                print(f"Successfully loaded with {method_name}")
                print(f"Type of data: {type(data)}")
                
                if isinstance(data, dict):
                    print(f"Keys: {data.keys()}")
                    print_dict_info(data)
                elif hasattr(data, 'columns') and hasattr(data, 'head'):  # DataFrame
                    print(f"DataFrame columns: {data.columns}")
                    print(f"DataFrame head:\n{data.head()}")
                else:
                    print(f"Data is not a dictionary or DataFrame. Value: {data}")
                
                break
            except Exception as e:
                print(f"Error with {method_name}: {str(e)}")
    
    except Exception as e:
        print(f"Error loading pickle file: {str(e)}")

if __name__ == "__main__":
    main()
