import os
import pickle
import numpy as np
from tqdm import tqdm

def read_pkl():
    pickle_path = os.path.dirname(os.path.realpath(__file__)) + '/train400_feature_all.pkl'
    if os.path.isfile(pickle_path):
        with open(pickle_path, 'rb') as handle:
            traj_dict = pickle.load(handle)
    else:
        raise FileNotFoundError("Pickle file not found")
    return traj_dict

def generate_pairs_and_labels(X_data, total_pairs=1000):
    half_pairs = total_pairs // 2  # Divide total_pairs to get half positive, half negative
    
    pos_pairs = []
    neg_pairs = []
    drivers = list(X_data.keys())
    
    # Generate Positive Pairs
    tqdm.write("Generating positive pairs...")
    while len(pos_pairs) < half_pairs:
        driver = np.random.choice(drivers)
        if len(X_data[driver]) < 2:  # Need at least 2 trajectories to form a pair
            continue
        days = np.random.choice(range(len(X_data[driver])), 2, replace=False)
        pos_pairs.append([X_data[driver][days[0]], X_data[driver][days[1]]])
    
    # Generate Negative Pairs
    tqdm.write("Generating negative pairs...")
    while len(neg_pairs) < half_pairs:
        driver1, driver2 = np.random.choice(drivers, 2, replace=False)
        if not X_data[driver1] or not X_data[driver2]:  # Skip if any driver has no data
            continue
        day1 = np.random.choice(range(len(X_data[driver1])))
        day2 = np.random.choice(range(len(X_data[driver2])))
        neg_pairs.append([X_data[driver1][day1], X_data[driver2][day2]])
    
    # Combine pairs and shuffle
    pairs = np.array(pos_pairs + neg_pairs)
    labels = np.array([1] * half_pairs + [0] * half_pairs)  # 1 for positive, 0 for negative
    
    return pairs, labels

def save_pairs_and_labels(pairs, labels, filename='X_Y_train400_pairs.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'pairs': pairs, 'labels': labels}, f)
    print(f"Saved pairs and labels to {filename}")

if __name__ == '__main__':
    X_data = read_pkl()
    total_pairs = 20000  # Total pairs you want to generate
    pairs, labels = generate_pairs_and_labels(X_data, total_pairs)
    
    # Ensure the shape is as expected
    print(f"Generated pairs shape: {pairs.shape}")
    print(f"Generated labels shape: {labels.shape}")
    
    save_pairs_and_labels(pairs, labels)