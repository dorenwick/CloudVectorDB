import os
import random
import re

import polars as pl
from pylatexenc.latex2text import LatexNodes2Text

latex = "Your LaTeX code here"
text = LatexNodes2Text().latex_to_text(latex)

# Traceback (most recent call last):
#   File "C:\Program Files\JetBrains\PyCharm Community Edition 2023.1.2\plugins\python-ce\helpers\pydev\pydevd.py", line 1496, in _exec
#     pydev_imports.execfile(file, globals, locals)  # execute the script
#   File "C:\Program Files\JetBrains\PyCharm Community Edition 2023.1.2\plugins\python-ce\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
#     exec(compile(contents+"\n", file, 'exec'), glob, loc)
#   File "C:\Users\doren\PycharmProjects\CloudVectorDB\AnalyseNgramFiles.py", line 160, in <module>
#     analyzer.analyze_all()
#   File "C:\Users\doren\PycharmProjects\CloudVectorDB\AnalyseNgramFiles.py", line 155, in analyze_all
#     self.normalize_ngrams()
#   File "C:\Users\doren\PycharmProjects\CloudVectorDB\AnalyseNgramFiles.py", line 135, in normalize_ngrams
#     df = normalized_ngrams.select([
#   File "C:\Users\doren\.conda\envs\CloudVectorDB\lib\site-packages\polars\dataframe\frame.py", line 8968, in select
#     return self.lazy().select(*exprs, **named_exprs).collect(_eager=True)
#   File "C:\Users\doren\.conda\envs\CloudVectorDB\lib\site-packages\polars\lazyframe\frame.py", line 2032, in collect
#     return wrap_df(ldf.collect(callback))
# polars.exceptions.SchemaError: invalid series dtype: expected `FixedSizeList`, got `str`



class AnalyzeNgramsFiles:
    """

    TODO: When we design a system for extracting keywords from an abstract to modify our meta-data in abstract-data, we shall do the following:
        we will create tables for the subfield, and the topics.

    Then, when we extract a keyword, we add it to the subfield count vector, and the topics count vector (around 250~ and 4500~ dimensions respectively,
        which shall be in columns of type list(int).
    After we have processed that table, (which should be around dimensions 1,000,000 multiplied by 5000,
    We will recompute additional scores.


    TODO: We dont actually wanna remove the field_count, smoothed_score, ctf_idf_score and other such fields.

    TODO: We shall need to fill in the scoring system.
        We shall also have to fill in the


    In a previous version, we removed html and latex:

    # Remove HTML entities
    unescaped = unescape(ngram)
    if unescaped != ngram and unescaped in ngram_dict and ngram_dict[unescaped] > count:
        normalized = unescaped
        count = ngram_dict[unescaped]

    # Remove LaTeX commands
    unlatexed = LatexNodes2Text().latex_to_text(ngram)
    if unlatexed != ngram and unlatexed in ngram_dict and ngram_dict[unlatexed] > count:
        normalized = unlatexed
        count = ngram_dict[unlatexed]


    """

    def __init__(self, directory=r"E:\NGRAMS"):
        self.directory = directory
        self.files = [
            "short_bigrams.parquet",
            "short_unigrams.parquet",
            "full_string_bigrams.parquet",
            "full_string_unigrams.parquet",
        ]
        self.dataframes = {}

    def load_dataframes(self):
        for file in self.files:
            file_path = os.path.join(self.directory, file)
            df = pl.read_parquet(file_path)
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
            df = df.filter(pl.col("count") >= 2)
            print(f"Length after filtering: {len(df)}")

            # Ensure all columns are of string type
            df = df.with_columns([
                pl.col("ngram").cast(pl.Utf8).alias("ngram"),
                pl.col("count").cast(pl.Int64).cast(pl.Utf8).alias("count")
            ])

            # Verify data types
            print(df.dtypes)

            # Create a dictionary for faster lookups
            ngram_dict = dict(zip(df['ngram'].to_list(), df['count'].to_list()))

            def normalize_single_ngram(ngram, count):
                if random.random() < 0.00001:
                    print("hi 100_000 processed")
                normalized = ngram
                original_count = count

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

                # Remove parentheses and their contents
                no_parentheses = re.sub(r'\([^)]*\)', '', ngram).strip()
                if no_parentheses != ngram and no_parentheses in ngram_dict and ngram_dict[no_parentheses] > count:
                    normalized = no_parentheses
                    count = ngram_dict[no_parentheses]

                # Remove common prefixes (e.g., "the", "a", "an")
                common_prefixes = ["the ", "a ", "an "]
                for prefix in common_prefixes:
                    if ngram.lower().startswith(prefix):
                        without_prefix = ngram[len(prefix):].strip()
                        if without_prefix in ngram_dict and ngram_dict[without_prefix] > count:
                            normalized = without_prefix
                            count = ngram_dict[without_prefix]

                return normalized if normalized != ngram else None, count

            # Apply normalization to each ngram
            normalized_ngrams = df.with_columns([
                pl.struct(['ngram', 'count']).map_elements(lambda struct: normalize_single_ngram(struct['ngram'], struct['count'])).alias('normalized')

            ])
            # , return_dtype=pl.String
            # Update the dataframe with normalized ngrams
            df = normalized_ngrams.select([
                pl.col('normalized').arr.get(0).alias('ngram'),
                pl.col('normalized').arr.get(1).alias('count')
            ]).filter(pl.col('ngram').is_not_null())

            # Convert count back to integers
            df = df.with_columns([
                pl.col('count').cast(pl.Int64)
            ])

            # Verify data types
            print("Final data types:")
            print(df.dtypes)

            # Save the normalized dataframe
            df.write_parquet(os.path.join(self.directory, f"normalized_{file}"))
            print(f"Saved normalized {file}")

    def analyze_all(self):
        self.load_dataframes()
        self.normalize_ngrams()


if __name__ == "__main__":
    analyzer = AnalyzeNgramsFiles()
    analyzer.analyze_all()
