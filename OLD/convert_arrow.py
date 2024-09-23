import polars as pl
import os

# Input file path
input_file = r"E:\data_backup\works_common_authors.parquet"

# Output directory
output_dir = r"E:\HugeDatasetBackup\cloud_datasets"

# Output file name
output_file = "works_common_authors.parquet"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Use Polars to lazily read the Parquet file and select the first 50,000 rows
df_lazy = pl.scan_parquet(input_file).limit(20000)

# Collect the result (this will only materialize the first 50,000 rows)
df_subset = df_lazy.collect()

# Save the subset as a new Parquet file
output_path = os.path.join(output_dir, output_file)
df_subset.write_parquet(output_path)

print(f"Saved {len(df_subset)} rows to {output_path}")