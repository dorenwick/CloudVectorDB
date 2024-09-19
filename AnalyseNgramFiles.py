import os
import random
import re
import time
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text

class AnalyzeNgramsFiles:


    def __init__(self, directory=r"E:\NGRAMS"):
        self.directory = directory
        self.files = [
            "short_bigrams.parquet",
            # "short_unigrams.parquet",
            # "full_string_bigrams.parquet",
            # "full_string_unigrams.parquet",
        ]
        self.dataframes = {}

    def load_dataframes(self):
        for file in self.files:
            file_path = os.path.join(self.directory, file)
            df = pd.read_parquet(file_path)
            self.dataframes[file] = df
            print(f"\nLoaded {file}")
            print(df.head(20))
            print(df.tail(20))
            print(f"Length of dataframe: {len(df)}")

    def normalize_ngrams(self):
        for file, df in self.dataframes.items():
            print(f"\nNormalizing {file}")
            print(f"Original length: {len(df)}")

            # Remove ngrams with count < 2
            df = df[df['count'] >= 2]
            print(f"Length after filtering: {len(df)}")

            # Ensure correct data types
            df['ngram'] = df['ngram'].astype(str)
            df['count'] = df['count'].astype(int)

            # Verify data types
            print("Data types:")
            print(df.dtypes)

            # Create a dictionary for faster lookups
            ngram_dict = dict(zip(df['ngram'], df['count']))
            print("DONE!")

            time.sleep(5)

            def normalize_single_ngram(row):
                ngram, count = row['ngram'], row['count']
                if random.random() < 0.00001:
                    print("hi 100_000 processed")
                normalized = ngram
                original_count = count
                original_type = type(ngram)

                # Strip non-alphanumeric characters
                stripped = re.sub(r'[^a-zA-Z0-9\s]', ' ', ngram).strip()
                stripped = re.sub(r'\s+', ' ', stripped)  # Remove extra spaces
                if stripped in ngram_dict and ngram_dict[stripped] > count:
                    normalized = stripped
                    count = ngram_dict[stripped]

                # Check for hyphenated bigrams
                if '-' in ngram and len(ngram.split()) == 2:
                    unhyphenated = ngram.replace('-', ' ')
                    if unhyphenated in ngram_dict and ngram_dict[unhyphenated] > count:
                        normalized = unhyphenated
                        count = ngram_dict[unhyphenated]

                # Remove trailing 's' if it's not a double 's'
                if ngram.endswith('s') and not ngram.endswith('ss'):
                    singular = ngram[:-1]
                    if singular in ngram_dict and ngram_dict[singular] > count:
                        normalized = singular
                        count = ngram_dict[singular]

                # Lowercase if it has a higher count
                lowercased = ngram.lower()
                if lowercased != ngram and lowercased in ngram_dict and ngram_dict[lowercased] > count:
                    normalized = lowercased
                    count = ngram_dict[lowercased]

                # Type check after normalization
                if type(normalized) != original_type:
                    print(f"Type changed for ngram: '{ngram}' (original) -> '{normalized}' (normalized)")
                    print(f"Original type: {original_type}, New type: {type(normalized)}")

                return pd.Series([normalized if normalized != ngram else None, count])

            # Apply normalization
            df[['normalized_ngram', 'normalized_count']] = df.apply(normalize_single_ngram, axis=1)

            # Update the dataframe with normalized ngrams
            df = df[df['normalized_ngram'].notnull()][['normalized_ngram', 'normalized_count']]
            df.columns = ['ngram', 'count']

            # Convert count back to integers
            df['count'] = df['count'].astype(int)

            # Verify data types
            print("Final data types:")
            print(df.dtypes)

            # Save the normalized dataframe
            df.to_parquet(os.path.join(self.directory, f"normalized_{file}"))
            print(f"Saved normalized {file}")

    def analyze_all(self):
        self.load_dataframes()
        self.normalize_ngrams()

if __name__ == "__main__":
    analyzer = AnalyzeNgramsFiles()
    analyzer.analyze_all()