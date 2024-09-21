import pandas as pd
import os

# Input and output file paths
input_file = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\datasets_collected\triplets_filtered_cleaned_truncated_hard.parquet"
output_directory = r"E:\HugeDatasetBackup\cloud_datasets"
output_file = os.path.join(output_directory, "triplets_test.parquet")

# Load the parquet file
print("Loading the parquet file...")
df = pd.read_parquet(input_file)

# Shuffle the dataframe
print("Shuffling the data...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Truncate to the top 100,000 rows
print("Truncating to the top 100,000 rows...")
df_truncated = df.head(100_000)

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Save the truncated dataframe as a new parquet file
print(f"Saving truncated data to {output_file}...")
df_truncated.to_parquet(output_file, index=False)

print("Done!")