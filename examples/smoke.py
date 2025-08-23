from urllib.request import urlretrieve
import zipfile
import os
import re
import pandas as pd


name = "smoke_detection_iot"
zip_filename = name + ".zip"
csv_filename = name + ".csv"
parquet_filename = name + ".parquet"

# Download the smoke detection dataset zip
url = ("https://www.kaggle.com/api/v1/datasets/download/deepcontractor/smoke-detection-dataset")
urlretrieve(url, zip_filename)

# Extract the zip
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall()

# Delete zip
os.remove(zip_filename)

# Read csv
with open(csv_filename, 'r') as file:
    csv_content = file.read()

# Remove row indices
csv_content_mod = re.sub('^[0-9]*,', '', csv_content)
csv_content_mod = re.sub('\n[0-9]*,', '\n', csv_content_mod)

# Save as csv
with open(csv_filename, 'w') as f:
    f.write(csv_content_mod)
    
# Convert to parquet
df = pd.read_csv(csv_filename)
df.to_parquet(parquet_filename, index=False)

# Delete csv
os.remove(csv_filename)

# Load the dataset
df = pd.read_parquet(parquet_filename)

# Shuffle since labels are very grouped
df = df.sample(frac=1)

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

# Delete full dataset
os.remove(parquet_filename)