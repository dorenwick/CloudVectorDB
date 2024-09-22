import os
import random
import re
import time
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text

class AnalyzeNgramsFiles:
    """
    Consider this class:
        The unigram and bigram files are actually too big to load into memory. What I want to do is a sort of pre-filtering
        system. We will load up the short and full bigrams in chunks of 1_000_000 rows at a time, and filter out the
        rows with count < 5. Then, we will save the chunks. Then, we will load up these filtered chunks and concatenate them
        into a single bigram file once again. Save them as filtered_short_bigrams.parquet and filtered_full_bigrams.parquet
        for us.

    TODO:



    """


    # Connected to pydev debugger (build 231.9414.12)
    # Loaded short_bigrams.parquet
    #                   ngram   count  ... smoothed_score  ctf_idf_score
    # 0     anomalous cooling       1  ...            0.0            0.0
    # 1            cooling of     436  ...            0.0            0.0
    # 2                of the  599733  ...            0.0            0.0
    # 3          the parallel     253  ...            0.0            0.0
    # 4     parallel velocity       2  ...            0.0            0.0
    # 5           velocity in     514  ...            0.0            0.0
    # 6             in seeded      13  ...            0.0            0.0
    # 7          seeded beams       2  ...            0.0            0.0
    # 8           beams alain       1  ...            0.0            0.0
    # 9          alain miffre       2  ...            0.0            0.0
    # 10        miffre marion       1  ...            0.0            0.0
    # 11       marion jacquey       2  ...            0.0            0.0
    # 12     jacquey matthias       1  ...            0.0            0.0
    # 13     matthias büchner       4  ...            0.0            0.0
    # 14       büchner gérard       1  ...            0.0            0.0
    # 15        gérard trénec       1  ...            0.0            0.0
    # 16            trénec j.       2  ...            0.0            0.0
    # 17             j. vigué       3  ...            0.0            0.0
    # 18  comparative genomic     287  ...            0.0            0.0
    # 19     genomic analysis     333  ...            0.0            0.0
    # [20 rows x 5 columns]
    #                           ngram  count  ... smoothed_score  ctf_idf_score
    # 52618832              p-c. sung      1  ...            0.0            0.0
    # 52618833      progressive cough      1  ...            0.0            0.0
    # 52618834  lymphocytic leukemoid      1  ...            0.0            0.0


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