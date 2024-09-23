import os
import shutil
import multiprocessing as mp
from math import ceil
import polars as pl
from itertools import chain


class KeyPhraseConstructor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_cores = mp.cpu_count()
        self.num_subdirs = ceil((self.num_cores / 3) * 4)

    def create_subdirectories(self):
        for i in range(self.num_subdirs):
            subdir = os.path.join(self.output_dir, f"subdir_{i}")
            os.makedirs(subdir, exist_ok=True)

    def distribute_files(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.parquet')]
        for i, file in enumerate(files):
            src = os.path.join(self.input_dir, file)
            dst = os.path.join(self.output_dir, f"subdir_{i % self.num_subdirs}", file)
            shutil.move(src, dst)

    def load_ngram_sets(self):
        ngram_sets = []
        for gram in ['uni', 'bi', 'tri', 'four']:
            file_path = os.path.join(self.output_dir, f"filtered_medium_Full{gram.capitalize()}gramProcessor.parquet")
            df = pl.read_parquet(file_path)
            ngram_sets.append(set(df['ngram'].to_list()))
        return ngram_sets

    def process_directory(self, subdir):
        ngram_sets = self.load_ngram_sets()

        for file in os.listdir(subdir):
            if file.endswith('.parquet'):
                file_path = os.path.join(subdir, file)
                df = pl.read_parquet(file_path)

                def extract_keyphrases(text):
                    words = text.lower().split()
                    unigrams = set(words)
                    bigrams = set(' '.join(words[i:i + 2]) for i in range(len(words) - 1))
                    trigrams = set(' '.join(words[i:i + 3]) for i in range(len(words) - 2))
                    fourgrams = set(' '.join(words[i:i + 4]) for i in range(len(words) - 3))

                    keyphrases = []
                    for gram_set, ngram_set in zip([unigrams, bigrams, trigrams, fourgrams], ngram_sets):
                        keyphrases.extend(gram_set.intersection(ngram_set))

                    return keyphrases

                df = df.with_columns(
                    pl.col('abstract_string').map_elements(extract_keyphrases).alias('full_keyphrases')
                )

                df.write_parquet(file_path)

    def run(self):
        self.create_subdirectories()
        self.distribute_files()

        subdirs = [os.path.join(self.output_dir, f"subdir_{i}") for i in range(self.num_subdirs)]

        with mp.Pool(processes=self.num_cores) as pool:
            pool.map(self.process_directory, subdirs)