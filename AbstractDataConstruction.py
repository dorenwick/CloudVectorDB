import json
import multiprocessing as mp
import os
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper


class AbstractDataConstruction:
    def __init__(self, input_dir: str, output_dir: str, batch_size: int = 100_000,
                 extract_keywords: bool = True, generate_embeddings: bool = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.extract_keywords = extract_keywords
        self.generate_embeddings_bool = generate_embeddings
        self.field_int_map = self.load_or_create_field_int_map()
        print("self.field_int_map: ", self.field_int_map)
        self.full_unigrams = Counter()
        self.full_bigrams = Counter()
        self.short_unigrams = Counter()
        self.short_bigrams = Counter()

    def load_or_create_field_int_map(self) -> Dict[str, Dict[str, int]]:
        field_int_map_path = os.path.join(self.output_dir, "field_int_map.json")
        if os.path.exists(field_int_map_path):
            with open(field_int_map_path, 'r') as f:
                return json.load(f)
        else:
            id2label = {
                0: "Economics, Econometrics and Finance", 1: "Materials Science",
                2: "Environmental Science", 3: "Medicine", 4: "Psychology", 5: "Dentistry",
                6: "Business, Management and Accounting", 7: "Engineering", 8: "Biochemistry, Genetics and Molecular Biology",
                9: "Agricultural and Biological Sciences", 10: "Energy", 11: "Earth and Planetary Sciences",
                12: "Health Professions", 13: "Chemistry", 14: "Chemical Engineering",
                15: "Social Sciences", 16: "Pharmacology, Toxicology and Pharmaceutics",
                17: "Arts and Humanities", 18: "Mathematics", 19: "Immunology and Microbiology",
                20: "Veterinary", 21: "Decision Sciences", 22: "Nursing",
                23: "Physics and Astronomy", 24: "Neuroscience", 25: "Computer Science"
            }
            label2id = {v: k for k, v in id2label.items()}
            field_int_map = {"id2label": id2label, "label2id": label2id}
            with open(field_int_map_path, 'w') as f:
                json.dump(field_int_map, f)
            return field_int_map

    def update_ngram_counters(self, df: pd.DataFrame):
        full_text = df['title'] + ' ' + df['authors_string'] + ' ' + df['abstract_string']
        short_text = df['title'] + ' ' + df['authors_string']

        with mp.Pool(processes=mp.cpu_count()) as pool:
            full_unigrams, full_bigrams = zip(*pool.map(self.process_ngrams, [(text, 1) for text in full_text]))
            short_unigrams, short_bigrams = zip(*pool.map(self.process_ngrams, [(text, 1) for text in short_text]))

        self.full_unigrams.update(sum(full_unigrams, Counter()))
        self.full_bigrams.update(sum(full_bigrams, Counter()))
        self.short_unigrams.update(sum(short_unigrams, Counter()))
        self.short_bigrams.update(sum(short_bigrams, Counter()))

    def process_ngrams(self, text_tuple: Tuple[str, int]) -> Tuple[Counter, Counter]:
        text, n = text_tuple
        unigrams = Counter(text.split())  # Example of how to split text into unigrams
        bigrams = Counter(zip(text.split()[:-1], text.split()[1:]))  # Example of bigrams
        return unigrams, bigrams

    def save_ngram_data(self):
        ngram_data = [
            (self.full_unigrams, "full_string_unigrams.parquet"),
            (self.full_bigrams, "full_string_bigrams.parquet"),
            (self.short_unigrams, "short_unigrams.parquet"),
            (self.short_bigrams, "short_bigrams.parquet")
        ]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(self.save_counter, [(counter, file_name, self.output_dir) for counter, file_name in ngram_data])

    @staticmethod
    def save_counter(counter: Counter, file_name: str, output_dir: str):
        df = pd.DataFrame.from_dict(counter, orient='index', columns=['count']).reset_index()
        df.columns = ['ngram', 'count']
        df['smoothed_score'] = 0.0
        df['ctf_idf_score'] = 0.0
        df['field_count'] = [np.zeros(26, dtype=int)] * len(df)
        df.to_parquet(os.path.join(output_dir, file_name), index=False)

    def clean_ngrams(self, df: pd.DataFrame) -> pd.DataFrame:
        def is_valid_ngram(ngram: str) -> bool:
            non_alpha_count = sum(1 for char in ngram if not char.isalpha() and char not in ["'", '"', '.', '$'])
            return non_alpha_count < 2

        mask = (df['count'] > 1) | ((df['count'] == 1) & (df['ngram'].apply(is_valid_ngram)))
        return df[mask]

    def post_process_ngram_data(self):
        file_names = ['full_string_unigrams.parquet', 'full_string_bigrams.parquet',
                      'short_unigrams.parquet', 'short_bigrams.parquet']

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(self.process_single_ngram_file, file_names)

        for file_name, df_cleaned in zip(file_names, results):
            output_path = os.path.join(self.output_dir, file_name)
            df_cleaned.to_parquet(output_path, index=False)

    def process_single_ngram_file(self, file_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.output_dir, file_name)
        df = pd.read_parquet(file_path)

        print(f"Processing {file_name}")
        print(f"Total rows before cleaning: {len(df)}")

        df_cleaned = self.clean_ngrams(df)
        print(f"Total rows after cleaning: {len(df_cleaned)}")

        df_cleaned['non_zero_count'] = df_cleaned['field_count'].apply(np.count_nonzero)
        B = 26  # Number of fields
        df_cleaned['ctf_idf_score'] = (B / (df_cleaned['non_zero_count'] + 1)) / np.log1p(df_cleaned['count'])

        return df_cleaned

    def process_ngrams(self):
        print("Starting ngram processing...")

        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])
        for file_name in tqdm(input_files, desc="Processing files for ngrams"):
            input_path = os.path.join(self.input_dir, file_name)
            df = pd.read_parquet(input_path)
            self.update_ngram_counters(df)
            print(f"Processed {file_name} for ngrams")

        print("Saving ngram data...")
        self.save_ngram_data()

        print("Post-processing ngram data...")
        self.post_process_ngram_data()

        print("Ngram processing completed successfully.")

    def run(self):
        self.process_ngrams()
        print("All processing completed successfully.")


if __name__ == "__main__":
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"

    processor = AbstractDataConstruction(
        input_dir=input_dir,
        output_dir=output_dir,
        extract_keywords=False,
        generate_embeddings=False
    )
    processor.run()
