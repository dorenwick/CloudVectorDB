import pandas as pd

# File path
file_path = r"E:\HugeDatasetBackup\ngram_mining_data\data\output\FullBigramProcessor.parquet"

# Read the Parquet file
df = pd.read_parquet(file_path)

# Print the first 100 rows
print(df.head(100).to_string())
print("length: ", len(df))


file_path = r"E:\HugeDatasetBackup\ngram_mining_data\data\output\ShortUnigramProcessor.parquet"


# Read the Parquet file
df = pd.read_parquet(file_path)

# Print the first 100 rows
print(df.head(100).to_string())
print("length: ", len(df))


file_path = r"E:\HugeDatasetBackup\ngram_mining_data\data\output\filtered_three_subfield_FullBigramProcessor.parquet"


# Read the Parquet file
df = pd.read_parquet(file_path)

# Print the first 100 rows
print(df.head(100).to_string())
print("length: ", len(df))

file_path = r"E:\HugeDatasetBackup\ngram_mining_data\data\output\filtered_three_subfield_FullUnigramProcessor.parquet"


# Read the Parquet file
df = pd.read_parquet(file_path)

# Print the first 100 rows
print(df.head(100).to_string())
print("length: ", len(df))



file_path = r"E:\HugeDatasetBackup\ngram_mining_data\data\output\filtered_two_field_ShortUnigramProcessor.parquet"


# Read the Parquet file
df = pd.read_parquet(file_path)

# Print the first 100 rows
print(df.head(100).to_string())
print("length: ", len(df))
