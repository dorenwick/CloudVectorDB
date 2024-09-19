import polars as pl
import os


def convert_arrow_to_parquet(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pl.read_ipc(input_file)

    print(f"Saving to {output_file}...")
    df.write_parquet(output_file)

    print(f"Conversion complete. Parquet file saved to {output_file}")


# File paths
unigram_arrow = r"E:\HugeDatasetBackup\cloud_ngrams\unigram_data.arrow"
bigram_arrow = r"E:\HugeDatasetBackup\cloud_ngrams\bigram_data.arrow"

unigram_parquet = r"E:\HugeDatasetBackup\cloud_ngrams\unigram_data.parquet"
bigram_parquet = r"E:\HugeDatasetBackup\cloud_ngrams\bigram_data.parquet"

# Convert unigram data
convert_arrow_to_parquet(unigram_arrow, unigram_parquet)

# Convert bigram data
convert_arrow_to_parquet(bigram_arrow, bigram_parquet)

print("All conversions completed.")