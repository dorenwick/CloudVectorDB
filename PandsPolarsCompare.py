import os
import psutil
import pandas as pd
import polars as pl
import time
from tqdm import tqdm



class PandasPolarsComparison:
    def __init__(self, pandas_file, polars_file, polars_streaming_file):
        self.pandas_file = pandas_file
        self.polars_file = polars_file
        self.polars_streaming_file = polars_streaming_file

    def print_memory_usage(self, location):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage at {location}: {memory_info.rss / 1024 / 1024:.2f} MB")

    def measure_dataframe_creation(self):
        print("\nMeasuring DataFrame Creation:")

        for df_type, file in [("Pandas", self.pandas_file),
                              ("Polars", self.polars_file),
                              ("Polars Streaming", self.polars_streaming_file)]:
            self.print_memory_usage(f"Before {df_type} DataFrame creation")
            start_time = time.time()

            if df_type.startswith("Polars"):
                df = pl.read_parquet(file)
            else:
                df = pd.read_parquet(file)

            end_time = time.time()
            self.print_memory_usage(f"After {df_type} DataFrame creation")
            print(f"{df_type} DataFrame creation time: {end_time - start_time:.2f} seconds")
            print(f"{df_type} DataFrame shape: {df.shape}")

            # Clear memory
            del df

    def measure_dataframe_loading(self):
        print("\nMeasuring DataFrame Loading:")

        for df_type, file in [("Pandas", self.pandas_file),
                              ("Polars", self.polars_file),
                              ("Polars Streaming", self.polars_streaming_file)]:
            self.print_memory_usage(f"Before {df_type} DataFrame loading")
            start_time = time.time()

            if df_type.startswith("Polars"):
                df = pl.scan_parquet(file).collect()
            else:
                df = pd.read_parquet(file)

            end_time = time.time()
            self.print_memory_usage(f"After {df_type} DataFrame loading")
            print(f"{df_type} DataFrame loading time: {end_time - start_time:.2f} seconds")
            print(f"{df_type} DataFrame shape: {df.shape}")

            # Clear memory
            del df

    def run_comparison(self):
        print("Starting Pandas/Polars Comparison")
        self.measure_dataframe_creation()
        self.measure_dataframe_loading()
        print("Comparison completed")


# Usage
if __name__ == "__main__":
    pandas_file = "E:\\HugeDatasetBackup\\cloud_datasets\\works_all_collected_pandas.parquet"
    polars_file = "E:\\HugeDatasetBackup\\cloud_datasets\\works_all_collected.parquet"
    polars_streaming_file = "E:\\HugeDatasetBackup\\cloud_datasets\\works_all_collected_polars_streaming.parquet"

    comparison = PandasPolarsComparison(pandas_file, polars_file, polars_streaming_file)
    comparison.run_comparison()