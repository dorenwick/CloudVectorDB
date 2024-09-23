import json
import os
import re
from collections import defaultdict
from typing import Dict
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm


def process_ngrams(df, ngram_type, field_int_map):
    ngrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})

    for _, row in df.iterrows():
        field = row['field']
        field_index = field_int_map['label2id'].get(field, -1)

        if ngram_type.startswith('full'):
            text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
        else:
            text = f"{row['title']} {row['authors_string']}".lower()

        words = text.split()

        if ngram_type.endswith('unigrams'):
            for word in words:
                if word.isalpha():
                    ngrams[word]['count'] += 1
                    if field_index != -1:
                        ngrams[word]['field_count'][field_index] += 1
        elif ngram_type.endswith('bigrams'):
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                if all(word.isalpha() for word in bigram.split()):
                    ngrams[bigram]['count'] += 1
                    if field_index != -1:
                        ngrams[bigram]['field_count'][field_index] += 1
        elif ngram_type.endswith('trigrams'):
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                if all(word.isalpha() for word in trigram.split()):
                    ngrams[trigram]['count'] += 1
                    if field_index != -1:
                        ngrams[trigram]['field_count'][field_index] += 1

    return ngrams


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

        self.ngram_types = [
            'full_unigrams', 'full_bigrams', 'full_trigrams',
            'short_unigrams', 'short_bigrams', 'short_trigrams'
        ]
        self.ngram_dicts = {ngram_type: defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
                            for ngram_type in self.ngram_types}

    def update_ngram_counters(self, df: pd.DataFrame):
        print("Lengths before update:")
        for ngram_type in self.ngram_types:
            print(f"{ngram_type}: {len(self.ngram_dicts[ngram_type])}")

        with mp.Pool(processes=6) as pool:
            results = [pool.apply_async(process_ngrams, (df, ngram_type, self.field_int_map))
                       for ngram_type in self.ngram_types]

            for ngram_type, result in zip(self.ngram_types, results):
                new_ngrams = result.get()
                for ngram, data in new_ngrams.items():
                    self.ngram_dicts[ngram_type][ngram]['count'] += data['count']
                    self.ngram_dicts[ngram_type][ngram]['field_count'] += data['field_count']

        print("Lengths after update:")
        for ngram_type in self.ngram_types:
            print(f"{ngram_type}: {len(self.ngram_dicts[ngram_type])}")

    def save_ngram_data(self):
        def save_counter(counter, file_name: str):
            df = pd.DataFrame([
                {'ngram': k, 'count': v['count'], 'field_count': v['field_count'].tolist()}
                for k, v in counter.items()
            ])
            df.to_parquet(os.path.join(self.output_dir, file_name), index=False)

        for ngram_type in self.ngram_types:
            save_counter(self.ngram_dicts[ngram_type], f"{ngram_type}.parquet")

        print(f"N-gram data saved. Current counter sizes:")
        for ngram_type in self.ngram_types:
            print(f"{ngram_type}: {len(self.ngram_dicts[ngram_type])}")

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])

        for file_name in tqdm(input_files, desc="Processing files"):
            try:
                input_path = os.path.join(self.input_dir, file_name)

                df = pd.read_parquet(input_path)
                self.update_ngram_counters(df)

                print(f"Processed {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        # Save n-gram data only once, at the end of processing
        self.save_ngram_data()
        print("All files processed successfully.")


    def run(self):
        print("Checking for missing parquet files and verifying row counts...")
        self.check_files_and_row_counts(check_row_consistency=False)

        print("\nStarting file processing...")
        self.process_files()
        print("Files processed.")
        print("Data processing completed successfully.")



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

if __name__ == "__main__":
    processor = AbstractDataConstructionMultiGPUOnly(is_local=True)  # Set to True for local testing
    processor.run()