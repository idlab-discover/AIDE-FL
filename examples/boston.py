import pandas as pd
from keras.datasets import boston_housing
import numpy as np
import os

name = "boston"
parquet_filename = name + ".parquet"

# Load Boston Housing dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Extract the RM (average number of rooms), CRIM (per capita crime rate by town), and INDUS (proportion of non-retail business acres per town) features
X_train_3d = X_train[:, [5,0,2]]
X_test_3d = X_test[:, [5,0,2]]

# Combine train and test sets
X = np.concatenate([X_train_3d, X_test_3d], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# Create DataFrame
feature_names_3d = ['RM', 'CRIM', 'INDUS']
df = pd.DataFrame(X, columns=feature_names_3d)
df['label'] = y

# Calculate the split size
n = len(df)
part_size = n // 7  # integer division

# Split into 3 parts
df1 = df.iloc[:part_size*3] # 3 parts
df2 = df.iloc[part_size*3:part_size*6] # 3 parts
df3 = df.iloc[6*part_size:] # 1 part for hold-out testing

# Save them back to parquet
df1.to_parquet(name + "_part1.parquet", index=False)
df2.to_parquet(name + "_part2.parquet", index=False)
df3.to_parquet(name + "_part3.parquet", index=False)