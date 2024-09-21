import pandas as pd

# Load the triplets_test.parquet file
triplets_file = r"E:\HugeDatasetBackup\cloud_datasets\triplets_test.parquet"
df = pd.read_parquet(triplets_file)

# Extract the first 100 rows and the required columns
test_sentences = df.head(5000)[['anchor_string', 'positive_string', 'negative_string']]

# Flatten the DataFrame to create a single column of sentences
test_sentences_flat = pd.DataFrame({
    'sentence': test_sentences.values.ravel()
})

# Save as test_sentences.parquet
output_file = r"E:\HugeDatasetBackup\cloud_datasets\test_sentences.parquet"
test_sentences_flat.to_parquet(output_file, index=False)

print(f"Created {output_file} with {len(test_sentences_flat)} sentences.")