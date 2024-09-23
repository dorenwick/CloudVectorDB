import json
import os
import re
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import re
from collections import defaultdict
from typing import Dict
import multiprocessing as mp

import numpy as np
import polars as pl
from tqdm import tqdm

class AbstractDataConstructionMultiGPUOnly:
    def __init__(self, is_local: bool = False, batch_size: int = 100_000):
        self.is_local = is_local
        self.batch_size = batch_size

        if self.is_local:
            self.input_dir = r"E:\HugeDatasetBackup\ngram_mining_data"
            self.output_dir = os.path.join(self.input_dir, "data", "output")
        else:
            self.input_dir = "/workspace"
            self.output_dir = "/workspace/data/output"

        self.field_int_map = self.load_or_create_field_int_map()
        print("self.field_int_map: ", self.field_int_map)

        self.full_unigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.full_bigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.full_trigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.short_unigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.short_bigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.short_trigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})


    def load_or_create_field_int_map(self) -> Dict[str, Dict[str, int]]:
        field_int_map_path = os.path.join(self.output_dir, "field_int_map.json")
        if os.path.exists(field_int_map_path):
            with open(field_int_map_path, 'r') as f:
                return json.load(f)
        else:
            # Create field_int_map based on the provided mapping
            id2label = {
                0: 'Biochemistry, Genetics and Molecular Biology',
                1: 'Engineering',
                2: 'Environmental Science',
                3: 'Mathematics',
                4: 'Social Sciences',
                5: 'Physics and Astronomy',
                6: 'Economics, Econometrics and Finance',
                7: 'Arts and Humanities',
                8: 'Chemistry',
                9: 'Agricultural and Biological Sciences',
                10: 'Medicine',
                11: 'Computer Science',
                12: 'Psychology',
                13: 'Chemical Engineering',
                14: 'Nursing',
                15: 'Pharmacology, Toxicology and Pharmaceutics',
                16: 'Business, Management and Accounting',
                17: 'Neuroscience',
                18: 'Materials Science',
                19: 'Health Professions',
                20: 'Immunology and Microbiology',
                21: 'Earth and Planetary Sciences',
                22: 'Energy',
                23: 'Dentistry',
                24: 'Veterinary',
                25: 'Decision Sciences'
            }
            label2id = {v: k for k, v in id2label.items()}
            field_int_map = {"id2label": id2label, "label2id": label2id}

            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            with open(field_int_map_path, 'w') as f:
                json.dump(field_int_map, f)
            return field_int_map

    def is_valid_ngram(self, ngram: str) -> bool:
        words = ngram.split()
        return len(words) <= 3 and all(word.isalpha() for word in words)

    def update_ngram_counters(self, df: pl.DataFrame):
        print("Lengths before update:")
        print(f"full_unigrams: {len(self.full_unigrams)}")
        print(f"short_unigrams: {len(self.short_unigrams)}")
        print(f"full_bigrams: {len(self.full_bigrams)}")
        print(f"short_bigrams: {len(self.short_bigrams)}")
        print(f"full_trigrams: {len(self.full_trigrams)}")
        print(f"short_trigrams: {len(self.short_trigrams)}")

        for row in df.iter_rows(named=True):
            field = row['field']
            field_index = self.field_int_map['label2id'].get(field, -1)
            full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
            short_text = f"{row['title']} {row['authors_string']}".lower()

            # Update unigrams, bigrams, and trigrams
            self.update_ngrams(full_text, self.full_unigrams, self.full_bigrams, self.full_trigrams, field_index)
            self.update_ngrams(short_text, self.short_unigrams, self.short_bigrams, self.short_trigrams, field_index)

        print("Lengths after update:")
        print(f"full_unigrams: {len(self.full_unigrams)}")
        print(f"short_unigrams: {len(self.short_unigrams)}")
        print(f"full_bigrams: {len(self.full_bigrams)}")
        print(f"short_bigrams: {len(self.short_bigrams)}")
        print(f"full_trigrams: {len(self.full_trigrams)}")
        print(f"short_trigrams: {len(self.short_trigrams)}")

    def update_ngrams(self, text, unigrams, bigrams, trigrams, field_index):
        words = text.split()
        for i in range(len(words)):
            if words[i].isalpha():
                unigrams[words[i]]['count'] += 1
                if field_index != -1:
                    unigrams[words[i]]['field_count'][field_index] += 1

            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if self.is_valid_ngram(bigram):
                    bigrams[bigram]['count'] += 1
                    if field_index != -1:
                        bigrams[bigram]['field_count'][field_index] += 1

            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if self.is_valid_ngram(trigram):
                    trigrams[trigram]['count'] += 1
                    if field_index != -1:
                        trigrams[trigram]['field_count'][field_index] += 1

    def save_ngram_data(self):
        def save_counter(counter, file_name: str):
            df = pl.DataFrame([
                {'ngram': k, 'count': v['count'], 'field_count': v['field_count'].tolist()}
                for k, v in counter.items()
            ])
            df.write_parquet(os.path.join(self.output_dir, file_name))

        save_counter(self.full_unigrams, "full_string_unigrams.parquet")
        save_counter(self.full_bigrams, "full_string_bigrams.parquet")
        save_counter(self.full_trigrams, "full_string_trigrams.parquet")
        save_counter(self.short_unigrams, "short_unigrams.parquet")
        save_counter(self.short_bigrams, "short_bigrams.parquet")
        save_counter(self.short_trigrams, "short_trigrams.parquet")

        print(f"N-gram data saved. Current counter sizes:")
        print(f"Full unigrams: {len(self.full_unigrams)}")
        print(f"Full bigrams: {len(self.full_bigrams)}")
        print(f"Full trigrams: {len(self.full_trigrams)}")
        print(f"Short unigrams: {len(self.short_unigrams)}")
        print(f"Short bigrams: {len(self.short_bigrams)}")
        print(f"Short trigrams: {len(self.short_trigrams)}")

    def process_file(self, file_name):
        try:
            input_path = os.path.join(self.input_dir, file_name)
            df = pl.read_parquet(input_path)
            self.update_ngram_counters(df)
            print(f"Processed {file_name}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])

        with mp.Pool(processes=mp.cpu_count()) as pool:
            list(tqdm(pool.imap(self.process_file, input_files), total=len(input_files), desc="Processing files"))

        self.save_ngram_data()
        print("All files processed successfully.")

    def check_files_and_row_counts(self, check_row_consistency=True):
        input_files = [f for f in os.listdir(self.input_dir) if
                       f.startswith('works_combined_data_batch_') and f.endswith('.parquet')]
        file_numbers = []

        for file in input_files:
            match = re.search(r'works_combined_data_batch_(\d+)\.parquet$', file)
            if match:
                file_numbers.append(int(match.group(1)))

        if not file_numbers:
            print("No matching parquet files found in the input directory.")
            return

        max_file_number = max(file_numbers)
        expected_numbers = set(range(max_file_number + 1))
        missing_numbers = expected_numbers - set(file_numbers)

        if missing_numbers:
            print(f"Missing batch numbers: {sorted(missing_numbers)}")
        else:
            print(f"All batch files from 0 to {max_file_number} are present.")

        if check_row_consistency:
            print("\nChecking row counts for each file:")
            for file_number in sorted(file_numbers):
                file_name = f"works_combined_data_batch_{file_number}.parquet"
                file_path = os.path.join(self.input_dir, file_name)
                try:
                    df = pd.read_parquet(file_path)
                    row_count = len(df)
                    if row_count != 100_000:
                        print(f"File {file_name} has {row_count} rows (expected 100,000).")
                    else:
                        print(f"File {file_name} has the correct number of rows (100,000).")
                except Exception as e:
                    print(f"Error reading file {file_name}: {str(e)}")

    def run(self):
        print("Checking for missing parquet files and verifying row counts...")
        self.check_files_and_row_counts(check_row_consistency=False)

        print("\nStarting file processing...")
        self.process_files()
        print("Files processed.")
        print("Data processing completed successfully.")

if __name__ == "__main__":
    processor = AbstractDataConstructionMultiGPUOnly(is_local=True)  # Set to True for local testing
    processor.run()