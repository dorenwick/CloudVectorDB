import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


class BaseNGramProcessor:
    """
    TODO: I wish for you to do the following: Add fourgram classes to this class.

    Also, once we finish every ngram parquet file, I want you to go through them and do a filtering of them.
    we will save f"filtered_{file_name}" parquet files of every version where we remove any ngram that doesn't have a 0 element in one of the vector
    fields, and doesn't have count greater or equal to 5.
    We will save them in the same location. Do this whenever we finish creating our ...NgramProcess.parquet files.
    Then, we will have filtered ones.

    After that again, we will save highly_filtered ngram files of the following type.

    we will save f"filtered_medium_{file_name}" files as well, where we create a stronger filter. We filter for ngrams that
    dont have count greater or equal to 5, and must have a 0 element in at least 15 entries for their field_count vector.

    So, do all this for me please.
    Also, Out of memory considerations, I do not want to start for four gram process until the shortunigram, shortbigram processes are done.
    Also, we will use a similar valid check as with the trigrams for the four grams.




    """



    def __init__(self, is_local: bool = False, batch_size: int = 100_000):
        self.is_local = is_local
        self.batch_size = batch_size
        self.input_dir = r"E:\HugeDatasetBackup\ngram_mining_data" if self.is_local else "/workspace"
        self.output_dir = os.path.join(self.input_dir, "data", "output")
        self.field_int_map = self.load_or_create_field_int_map()
        self.ngrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})

    def load_or_create_field_int_map(self):
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
            os.makedirs(self.output_dir, exist_ok=True)
            with open(field_int_map_path, 'w') as f:
                json.dump(field_int_map, f)
            return field_int_map

    def is_valid_ngram(self, ngram: str) -> bool:
        words = ngram.split()

        # Check if all words are alphabetic
        if all(word.isalpha() for word in words):
            return True

        # Check if all words except the last one are alphabetic,
        # and the last word is alphabetic except for its last character
        elif (all(word.isalpha() for word in words[:-1]) and
              words[-1][:-1].isalpha()):
            # Remove the last character from the ngram
            return ngram[:-1]

        return False

    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            field_index = self.field_int_map['label2id'].get(row['field'], -1)
            full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
            words = full_text.split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                valid_bigram = self.is_valid_ngram(bigram)
                if valid_bigram:
                    # If valid_bigram is True, use the original bigram
                    # If it's a string, use the returned modified bigram
                    ngram_to_use = bigram if valid_bigram is True else valid_bigram
                    self.ngrams[ngram_to_use]['count'] += 1
                    if field_index != -1:
                        self.ngrams[ngram_to_use]['field_count'][field_index] += 1

    def save_ngram_data(self):
        df = pd.DataFrame([
            {'ngram': k, 'count': v['count'], 'field_count': v['field_count'].tolist()}
            for k, v in self.ngrams.items()
        ])
        df.to_parquet(os.path.join(self.output_dir, f"{self.__class__.__name__}.parquet"), index=False)

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])
        save_intervals = [1, 10, 100, 1000]

        for i, file_name in enumerate(tqdm(input_files, desc=f"Processing {self.__class__.__name__}"), start=1):
            try:
                input_path = os.path.join(self.input_dir, file_name)
                df = pd.read_parquet(input_path)
                self.update_ngram_counters(df)

                # Save at specified intervals or every 1000 files after that
                if i in save_intervals or (i > 1000 and i % 1000 == 0):
                    self.save_ngram_data()
                    print(f"Saved ngram data after processing {i} files.")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        # Final save after processing all files
        self.save_ngram_data()
        print("Finished processing all files. Final ngram data saved.")

class FullUnigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            field_index = self.field_int_map['label2id'].get(row['field'], -1)
            full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
            for word in full_text.split():
                self.ngrams[word]['count'] += 1
                if field_index != -1:
                    self.ngrams[word]['field_count'][field_index] += 1

class ShortUnigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            field_index = self.field_int_map['label2id'].get(row['field'], -1)
            short_text = f"{row['title']} {row['authors_string']}".lower()
            for word in short_text.split():
                self.ngrams[word]['count'] += 1
                if field_index != -1:
                    self.ngrams[word]['field_count'][field_index] += 1

class FullBigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            field_index = self.field_int_map['label2id'].get(row['field'], -1)
            full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
            words = full_text.split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                if self.is_valid_ngram(bigram):
                    self.ngrams[bigram]['count'] += 1
                    if field_index != -1:
                        self.ngrams[bigram]['field_count'][field_index] += 1

class ShortBigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            field_index = self.field_int_map['label2id'].get(row['field'], -1)
            short_text = f"{row['title']} {row['authors_string']}".lower()
            words = short_text.split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                if self.is_valid_ngram(bigram):
                    self.ngrams[bigram]['count'] += 1
                    if field_index != -1:
                        self.ngrams[bigram]['field_count'][field_index] += 1

class FullTrigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            field_index = self.field_int_map['label2id'].get(row['field'], -1)
            full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
            words = full_text.split()
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                if self.is_valid_ngram(trigram):
                    self.ngrams[trigram]['count'] += 1
                    if field_index != -1:
                        self.ngrams[trigram]['field_count'][field_index] += 1

class ShortTrigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            field_index = self.field_int_map['label2id'].get(row['field'], -1)
            short_text = f"{row['title']} {row['authors_string']}".lower()
            words = short_text.split()
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                if self.is_valid_ngram(trigram):
                    self.ngrams[trigram]['count'] += 1
                    if field_index != -1:
                        self.ngrams[trigram]['field_count'][field_index] += 1

def run_processor(processor_class):
    processor = processor_class(is_local=True)
    processor.process_files()

if __name__ == "__main__":
    processors = [
        FullUnigramProcessor,
        ShortUnigramProcessor,
        FullBigramProcessor,
        ShortBigramProcessor,
        FullTrigramProcessor,
        ShortTrigramProcessor
    ]

    with mp.Pool(processes=min(mp.cpu_count(), len(processors))) as pool:
        pool.map(run_processor, processors)

    print("All processors completed successfully.")