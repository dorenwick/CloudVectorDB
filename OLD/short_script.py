import pandas as pl

# File path
file_path = r"E:\HugeDatasetBackup\cloud_vectordbs\works_id_mapping.parquet"

# Read the Parquet file
df = pl.read_parquet(file_path)

# Print the schema
print("Schema:")
print(df.describe())

# Print the number of rows
print(f"\nTotal number of rows: {len(df)}")

# Print the top 100 rows
print("\nTop 100 rows:")
print(df.head(100).to_string())

# Optional: If you want to see unique values in a specific column
# Assuming 'works_int_id' is the column you're interested in
print("\nUnique values in 'works_int_id' column:")
print(df['works_int_id'].unique().sort().head(100))