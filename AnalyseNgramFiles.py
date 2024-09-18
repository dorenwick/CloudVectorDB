import pandas as pd
import os


class AnalyzeNgramsFiles:
    """
    A class to analyze ngram files stored in parquet format.
    Files to analyze:
    """

    def __init__(self, directory=r"C:\Users\doren\PycharmProjects\CloudVectorDB"):
        self.directory = directory
        self.files = [
            "keywords_data.parquet",
            "full_string_unigrams.parquet",
            "short_bigrams.parquet",
            "short_unigrams.parquet"
        ]
        self.dataframes = {}

    def load_dataframes(self):
        for file in self.files:
            file_path = os.path.join(self.directory, file)
            df = pd.read_parquet(file_path)
            self.dataframes[file] = df
            print(f"\nLoaded {file}")

    def print_schema_details(self):
        for file, df in self.dataframes.items():
            print(f"\nSchema details for {file}:")
            print(df.info())

    def print_head_and_tail(self):
        for file, df in self.dataframes.items():
            print(f"\nFirst 100 rows of {file}:")
            print(df.head(100).to_string())
            print(f"\nLast 100 rows of {file}:")
            print(df.tail(100).to_string())

    def analyze_all(self):
        self.load_dataframes()
        self.print_schema_details()
        self.print_head_and_tail()


if __name__ == "__main__":
    analyzer = AnalyzeNgramsFiles()
    analyzer.analyze_all()