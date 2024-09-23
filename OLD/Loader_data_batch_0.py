import time

import pandas as pd
import polars as pl
import os

import psutil


class ParquetLoader:
    """
    5gb per 1 million rows = 500gb for 100 million rows, which means, 1.5TB for 300 million rows.
    However, the abstract_string will be removed in the future, for various purposes.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        self.print_memory_usage("before loading")
        self.df = pl.read_parquet(self.file_path)
        self.print_memory_usage("after loading")
        print(f"Loaded {self.file_path} successfully.")

        # Drop the abstract_string column
        if 'abstract_string' in self.df.columns:
            self.df = self.df.drop('abstract_string')
            print(self.df)
            self.print_memory_usage("after dropping abstract_string column")
        else:
            print("Note: 'abstract_string' column not found in the DataFrame.")

    def print_head_tail(self, n=20):
        if self.df is None:
            print("Please load the data first using load_data() method.")
            return

        print(f"\n--- First {n} rows ---")
        print(self.df.head(n).to_string())

        print(f"\n--- Last {n} rows ---")
        print(self.df.tail(n).to_string())

    def print_memory_usage(self, location):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage at {location}: {memory_info.rss / 1024 / 1024:.2f} MB")

    def print_details(self):
        if self.df is None:
            print("Please load the data first using load_data() method.")
            return

        print("\n--- DataFrame Info ---")
        print(self.df.schema)

        print("\n--- Descriptive Statistics ---")
        print(self.df.describe().to_string())

        print("\n--- Column Names ---")
        print(self.df.columns)

        print("\n--- Data Types ---")
        print(self.df.dtypes)

    def process(self):
        self.load_data()
        self.print_head_tail()
        self.print_details()


# Usage
if __name__ == "__main__":
    file_path = r"E:\HugeDatasetBackup\works_batch_0.parquet"
    loader = ParquetLoader(file_path)
    loader.process()