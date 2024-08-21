from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
from ..config import MAJORITY_FRAC, SAMPLING_FRAC, TARGET, UNSW_PATH, TON_PATH
import gc


def undersample(X_train, y_train, majority_reduction_factor=MAJORITY_FRAC):
    data_combined = pd.concat([X_train, y_train], axis=1)

    majority_class_label = y_train.value_counts().idxmax()
    minority_class_label = y_train.value_counts().idxmin()

    majority_class = data_combined[data_combined[y_train.name] == majority_class_label]
    minority_class = data_combined[data_combined[y_train.name] == minority_class_label]
    
    reduced_majority_size = int(len(majority_class) * majority_reduction_factor)
    
    new_majority_size = max(reduced_majority_size, len(minority_class))

    majority_class_downsampled = majority_class.sample(new_majority_size)
    
    data_balanced = pd.concat([minority_class, majority_class_downsampled], copy=False, sort=False)
    
    data_balanced = shuffle(data_balanced)
    
    X_train_balanced = data_balanced.drop(columns=[y_train.name])
    y_train_balanced = data_balanced[y_train.name]

    print("Finished undersampling")
    return X_train_balanced, y_train_balanced

def drop_percentile_outliers(X_train, y_train):
    mask = [True] * len(X_train)
    
    for column in X_train.select_dtypes(include=['float32', 'int32']).columns:
        low_percentile = X_train[column].quantile(0.05)
        high_percentile = X_train[column].quantile(0.95)
        
        column_mask = (X_train[column] >= low_percentile) & (X_train[column] <= high_percentile)
        mask = mask & column_mask    
    
    return X_train[mask], y_train[mask]

def preprocess_queensland_data(PATH_DATA, isTesting=False):
    data = pd.read_parquet(PATH_DATA)
        
    columns_to_exclude = ['Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Dataset', 'L4_SRC_PORT', 'L4_DST_PORT']
    data = data[[column for column in data.columns if column not in columns_to_exclude]]
    # data = data[(data >= 0).all(axis=1)]
        
    if(PATH_DATA != UNSW_PATH or PATH_DATA != TON_PATH): # Allow full data for UNSW as its on the low side
        data = data.sample(frac=SAMPLING_FRAC)
    
    print("Splitting")
    X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data[data.columns[-1]], test_size=0.2, stratify=data[data.columns[-1]])    
    del [[data]]
    gc.collect()
    
    print("Dropping edge cases: [0, 5%], [95%, 100%]")
    X_train, y_train = drop_percentile_outliers(X_train, y_train)

    print(f"Undersampling with majority fraction:\t{MAJORITY_FRAC}")
    X_train, y_train = undersample(X_train, y_train)
    
    if(len(X_train) > TARGET):
        print(f"Downscale to {TARGET} samples")
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=TARGET, stratify=y_train)

    print("Scaling")
    scaling_lookup_table = {}
    for c in X_train.columns:
        scaling_lookup_table[c] = (X_train[c].min(),X_train[c].max())            
        X_train[c] = (X_train[c] - scaling_lookup_table[c][0]) / (scaling_lookup_table[c][1] - scaling_lookup_table[c][0])        
    for c in X_test.columns:
        X_test[c] = (X_test[c] - scaling_lookup_table[c][0]) / (scaling_lookup_table[c][1] - scaling_lookup_table[c][0])
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)  
    print(f"Preprocessing created {len(X_train)} samples to fit on!!!")
    print(f"Preprocessing created {len(X_val)} samples to validate on!!!")
    print(f"Preprocessing created {len(X_test)} samples to test on!!!")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)