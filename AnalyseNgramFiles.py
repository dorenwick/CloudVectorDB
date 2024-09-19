import polars as pl
import os
import re
from html import unescape
from pylatexenc.latex2text import LatexNodes2Text

latex = "Your LaTeX code here"
text = LatexNodes2Text().latex_to_text(latex)


class AnalyzeNgramsFiles:
    def __init__(self, directory=r"E:\NGRAMS"):
        self.directory = directory
        self.files = [
            "full_string_bigrams.parquet",
            "full_string_unigrams.parquet",
            "short_bigrams.parquet",
            "short_unigrams.parquet",
        ]
        self.dataframes = {}

    def load_dataframes(self):
        for file in self.files:
            file_path = os.path.join(self.directory, file)
            df = pl.read_parquet(file_path)
            self.dataframes[file] = df
            print(f"\nLoaded {file}")
            print(f"Length of dataframe: {len(df)}")

    def normalize_ngrams(self):
        for file, df in self.dataframes.items():
            print(f"\nNormalizing {file}")

            # Remove ngrams with count < 5
            df = df.filter(pl.col("count") >= 5)

            # Create a dictionary for faster lookups
            ngram_dict = dict(zip(df['ngram'].to_list(), df['count'].to_list()))

            def normalize_batch(batch):
                ngrams = batch['ngram']
                counts = batch['count']

                def normalize_single_ngram(ngram, count):
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

                results = [normalize_single_ngram(ngram, count) for ngram, count in zip(ngrams, counts)]
                normalized_ngrams, new_counts = zip(*results)

                return pl.Series(name='normalized', values=[
                    {'ngram': ngram, 'count': count} if ngram is not None else None
                    for ngram, count in zip(normalized_ngrams, new_counts)
                ])

                # Apply normalization to each batch

            normalized_df = df.with_columns([
                pl.struct(['ngram', 'count']).map_batches(normalize_batch, return_dtype=pl.List(
                    pl.Struct([pl.Field('ngram', pl.Utf8), pl.Field('count', pl.Int64)])))
            ])

            # Explode the result and filter out None values
            normalized_df = normalized_df.explode('normalized').filter(pl.col('normalized').is_not_null())

            # Extract ngram and count from the struct
            normalized_df = normalized_df.with_columns([
                pl.col('normalized').struct.field('ngram').alias('ngram'),
                pl.col('normalized').struct.field('count').alias('count')
            ]).drop('normalized')

            # Save the normalized dataframe
            normalized_df.write_parquet(os.path.join(self.directory, f"normalized_{file}"))
            print(f"Saved normalized {file}")

    def analyze_all(self):
        self.load_dataframes()
        self.normalize_ngrams()


if __name__ == "__main__":
    analyzer = AnalyzeNgramsFiles()
    analyzer.analyze_all()
