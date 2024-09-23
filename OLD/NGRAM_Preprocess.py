import polars as pl
import os
from itertools import groupby




class KeywordGenerator:
    """

    We want between min_count and max_count, or has min_count and only appears
    in two fields at most.


    # "E:\NGRAMS\full_string_bigrams.parquet"
    # "E:\NGRAMS\full_string_unigrams.parquet"

    I want you to make a method that creates:

     filtered_full_string_bigrams.parquet
     filtered_full_string_unigrams.parquet



    """



    def __init__(self, ngrams_directory, batches_directory):
        self.ngrams_directory = ngrams_directory
        self.batches_directory = batches_directory
        self.unigrams = None
        self.bigrams = None

    def load_ngrams(self):
        unigrams_path = os.path.join(self.ngrams_directory, "filtered_full_string_unigrams.parquet")
        bigrams_path = os.path.join(self.ngrams_directory, "filtered_full_string_bigrams.parquet")

        self.unigrams = pl.scan_parquet(unigrams_path).filter(pl.col("count").is_between(10, 10_000)).collect()
        self.bigrams = pl.scan_parquet(bigrams_path).filter(pl.col("count").is_between(5, 5_000)).collect()

        self.unigrams_set = set(self.unigrams["ngram"].to_list())
        self.bigrams_set = set(self.bigrams["ngram"].to_list())

    def tokenize(self, text):
        words = text.lower().split()
        unigrams = words
        bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
        trigrams = [f"{words[i]} {words[i + 1]} {words[i + 2]}" for i in range(len(words) - 2)]
        fourgrams = [f"{words[i]} {words[i + 1]} {words[i + 2]} {words[i + 3]}" for i in range(len(words) - 3)]
        return unigrams, bigrams, trigrams, fourgrams

    def generate_keywords(self, abstract):
        unigrams, bigrams, trigrams, fourgrams = self.tokenize(abstract)

        word_flags = [1 if word in self.unigrams_set else 0 for word in unigrams]
        for i in range(len(unigrams) - 1):
            if f"{unigrams[i]} {unigrams[i + 1]}" in self.bigrams_set:
                word_flags[i] = word_flags[i + 1] = 1

        keywords = []
        for k, g in groupby(enumerate(word_flags), key=lambda x: x[1]):
            if k == 1:
                group = list(g)
                start = group[0][0]
                end = group[-1][0]
                keyword = " ".join(unigrams[start:end + 1])
                keywords.append(keyword)

        return keywords

    def process_batch(self, batch):
        return batch.with_columns([
            pl.col("abstract_string").map_elements(self.generate_keywords).alias("keywords")
        ])

    def process_batches(self):
        batch_files = [f for f in os.listdir(self.batches_directory) if f.endswith(".parquet")]

        for batch_file in batch_files:
            print(f"Processing {batch_file}")
            batch_path = os.path.join(self.batches_directory, batch_file)
            df = pl.read_parquet(batch_path)

            processed_df = self.process_batch(df)

            output_path = os.path.join(self.batches_directory, f"processed_{batch_file}")
            processed_df.write_parquet(output_path)
            print(f"Saved processed batch to {output_path}")

    def run(self):
        print("Loading ngrams...")
        self.load_ngrams()
        print("Processing batches...")
        self.process_batches()
        print("Keyword generation complete.")


if __name__ == "__main__":
    ngrams_dir = r"E:\NGRAMS"
    batches_dir = r"E:\DATASETS"
    generator = KeywordGenerator(ngrams_dir, batches_dir)
    generator.run()