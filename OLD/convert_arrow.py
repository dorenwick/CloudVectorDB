import pandas as pd

# File path
file_path = r"E:\HugeDatasetBackup\ngram_mining_data\data\output\ShortTrigramProcessor.parquet"

# Read the Parquet file
df = pd.read_parquet(file_path)

# Print the first 100 rows
print(df.head(1000).to_string())