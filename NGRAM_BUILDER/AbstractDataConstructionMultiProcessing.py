import gc
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from KeyPhraseConstructor import KeyPhraseConstructor


class BaseNGramProcessor:
    """
    python3 AbstractDataConstructionMultiprocessing.py

    TODO: We will implement a post processing step for our class here, which will go back, load up each parquet batch file,
        and then read the abstracts and place the ngrams into them.

        We will make another class for the post-processing step. This class shall be called after everything else is completed.
        What it will do is this.
        It will, calculate the number of cpu cores we have, and then create int((num_cpu cores / 3) * 4) many subdirectories
        and move (shutil) the batch files to those subdirectories.
        Then, we will assign a cpu core to each directory and go through all the batch files,

         and load up the:

        highly_filtered_output = os.path.join(os.path.dirname(output_file), f"filtered_medium_{os.path.basename(output_file)}")

        files for uni, bi, tri, and fourgrams.

        And we will go through all the batches and do this: We will get create full_keyphrase column, and
        then tokenize the abstract string into the 1,2,3,4 grams, and then use our lookup table of the ngrams on the 4 filtered medium
        parquet files, which we will turn their ngram columns into sets with string elements, (4 sets), and whenever our lookup
        table finds an ngram, we add it to the full_keyphrases column, which will be a list of all the ngrams we found for the abstract_string.

        Please use polars in this class, to implement it for us.

        #  python3 AbstractDataConstructionMultiProcessing.py


    TODO:
    """


    def __init__(self, is_local: bool = False, batch_size: int = 100_000):
        self.is_local = is_local
        self.is_local = False
        self.batch_size = batch_size
        self.input_dir = r"E:\HugeDatasetBackup\ngram_mining_data" if self.is_local else "/workspace"
        self.output_dir = os.path.join(self.input_dir, "data", "output")
        self.load_mappings()
        self.ngrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.cleanup_interval = self.get_cleanup_interval()

    def get_cleanup_interval(self):
        if isinstance(self, FullUnigramProcessor):
            return 50
        elif isinstance(self, FullBigramProcessor):
            return 40
        elif isinstance(self, FullTrigramProcessor):
            return 30
        elif isinstance(self, FullFourgramProcessor):
            return 10
        else:
            return None

    def load_mappings(self):
        # Load field_int_map
        self.field_int_map = self.load_or_create_field_int_map()

        # Load subfield_int_map
        subfield_int_map_path = os.path.join(self.output_dir, "subfield_int_map.json")
        if os.path.exists(subfield_int_map_path):
            with open(subfield_int_map_path, 'r') as f:
                self.subfield_int_map = json.load(f)
        else:
            print(f"Warning: subfield_int_map.json not found at {subfield_int_map_path}")
            self.subfield_int_map = None

        # Load topic_int_map
        topic_int_map_path = os.path.join(self.output_dir, "topic_int_map.json")
        if os.path.exists(topic_int_map_path):
            with open(topic_int_map_path, 'r') as f:
                self.topic_int_map = json.load(f)
        else:
            print(f"Warning: topic_int_map.json not found at {topic_int_map_path}")
            self.topic_int_map = None

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
        raise NotImplementedError("This method should be implemented in subclasses")

    def save_ngram_data(self):
        df = pd.DataFrame([
            {'ngram': k, 'count': v['count'], 'field_count': v['field_count'].tolist()}
            for k, v in self.ngrams.items()
        ])
        output_file = os.path.join(self.output_dir, f"{self.__class__.__name__}.parquet")
        df.to_parquet(output_file, index=False)
        return output_file

    def cleanup_ngrams(self):
        self.ngrams = {k: v for k, v in self.ngrams.items() if v['count'] > 1}
        self.save_ngram_data()

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])
        save_intervals = [1, 5, 10, 500, 1000]
        max_files = 15 if self.is_local else len(input_files)

        for i, file_name in enumerate(tqdm(input_files[:max_files], desc=f"Processing {self.__class__.__name__}"),
                                      start=1):
            try:
                input_path = os.path.join(self.input_dir, file_name)
                df = pd.read_parquet(input_path)
                self.update_ngram_counters(df)

                if self.cleanup_interval and i % self.cleanup_interval == 0:
                    self.cleanup_ngrams()

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        output_file = self.save_ngram_data()
        print("Finished processing all files. Final ngram data saved.")
        return output_file

def filter_ngrams(input_file, output_file, min_count=5, min_zero_fields=1):
    df = pd.read_parquet(input_file)
    df['zero_fields'] = df['field_count'].apply(lambda x: sum(np.array(x) == 0))
    filtered_df = df[(df['count'] >= min_count) & (df['zero_fields'] >= min_zero_fields)]
    filtered_df.drop('zero_fields', axis=1, inplace=True)
    filtered_df.to_parquet(output_file, index=False)

def filter_two_fields(input_file, output_file):
    df = pd.read_parquet(input_file)
    df['non_zero_fields'] = df['field_count'].apply(lambda x: sum(np.array(x) > 0))
    filtered_df = df[df['non_zero_fields'] == 2]
    filtered_df.drop('non_zero_fields', axis=1, inplace=True)
    filtered_df.to_parquet(output_file, index=False)

def filter_three_subfields(input_file, output_file, subfield_mapping):
    df = pd.read_parquet(input_file)

    def count_subfields(field_count):
        subfield_count = 0
        for field, count in enumerate(field_count):
            if count > 0:
                subfield_count += len(subfield_mapping[str(field)])
        return subfield_count

    df['subfield_count'] = df['field_count'].apply(count_subfields)
    filtered_df = df[df['subfield_count'] <= 3]
    filtered_df.drop('subfield_count', axis=1, inplace=True)
    filtered_df.to_parquet(output_file, index=False)



class FullUnigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
                for word in full_text.split():
                    if word not in self.ngrams:
                        self.ngrams[word] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                    self.ngrams[word]['count'] += 1
                    if field_index != -1:
                        self.ngrams[word]['field_count'][field_index] += 1
            except Exception as e:
                print(f"Error processing row in FullUnigramProcessor: {e}")

class ShortUnigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):

        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                short_text = f"{row['title']} {row['authors_string']}".lower()
                for word in short_text.split():
                    if word not in self.ngrams:
                        self.ngrams[word] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                    self.ngrams[word]['count'] += 1
                    if field_index != -1:
                        self.ngrams[word]['field_count'][field_index] += 1
            except Exception as e:
                print("row: ", row)
                print(f"Error processing row in ShortUnigramProcessor: {e}")
class FullBigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
                words = full_text.split()
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i + 1]}"
                    if self.is_valid_ngram(bigram):
                        if bigram not in self.ngrams:
                            self.ngrams[bigram] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                        self.ngrams[bigram]['count'] += 1
                        if field_index != -1:
                            self.ngrams[bigram]['field_count'][field_index] += 1
            except Exception as e:
                print(f"Error processing row in FullBigramProcessor: {e}")

class ShortBigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                short_text = f"{row['title']} {row['authors_string']}".lower()
                words = short_text.split()
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i + 1]}"
                    if self.is_valid_ngram(bigram):
                        if bigram not in self.ngrams:
                            self.ngrams[bigram] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                        self.ngrams[bigram]['count'] += 1
                        if field_index != -1:
                            self.ngrams[bigram]['field_count'][field_index] += 1
            except Exception as e:
                print(f"Error processing row in ShortBigramProcessor: {e}")

class FullTrigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
                words = full_text.split()
                for i in range(len(words) - 2):
                    trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                    valid_trigram = self.is_valid_ngram(trigram)
                    if valid_trigram:
                        ngram_to_use = trigram if valid_trigram is True else valid_trigram
                        if ngram_to_use not in self.ngrams:
                            self.ngrams[ngram_to_use] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                        self.ngrams[ngram_to_use]['count'] += 1
                        if field_index != -1:
                            self.ngrams[ngram_to_use]['field_count'][field_index] += 1
            except Exception as e:
                print(f"Error processing row in FullTrigramProcessor: {e}")

class ShortTrigramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                short_text = f"{row['title']} {row['authors_string']}".lower()
                words = short_text.split()
                for i in range(len(words) - 2):
                    trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                    valid_trigram = self.is_valid_ngram(trigram)
                    if valid_trigram:
                        ngram_to_use = trigram if valid_trigram is True else valid_trigram
                        if ngram_to_use not in self.ngrams:
                            self.ngrams[ngram_to_use] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                        self.ngrams[ngram_to_use]['count'] += 1
                        if field_index != -1:
                            self.ngrams[ngram_to_use]['field_count'][field_index] += 1
            except Exception as e:
                print(f"Error processing row in ShortTrigramProcessor: {e}")

class FullFourgramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                full_text = f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower()
                words = full_text.split()
                for i in range(len(words) - 3):
                    fourgram = f"{words[i]} {words[i + 1]} {words[i + 2]} {words[i + 3]}"
                    valid_fourgram = self.is_valid_ngram(fourgram)
                    if valid_fourgram:
                        ngram_to_use = fourgram if valid_fourgram is True else valid_fourgram
                        if ngram_to_use not in self.ngrams:
                            self.ngrams[ngram_to_use] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                        self.ngrams[ngram_to_use]['count'] += 1
                        if field_index != -1:
                            self.ngrams[ngram_to_use]['field_count'][field_index] += 1
            except Exception as e:
                print(f"Error processing row in FullFourgramProcessor: {e}")

class ShortFourgramProcessor(BaseNGramProcessor):
    def update_ngram_counters(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                field_index = self.field_int_map['label2id'].get(row['field'], -1)
                short_text = f"{row['title']} {row['authors_string']}".lower()
                words = short_text.split()
                for i in range(len(words) - 3):
                    fourgram = f"{words[i]} {words[i + 1]} {words[i + 2]} {words[i + 3]}"
                    valid_fourgram = self.is_valid_ngram(fourgram)
                    if valid_fourgram:
                        ngram_to_use = fourgram if valid_fourgram is True else valid_fourgram
                        if ngram_to_use not in self.ngrams:
                            self.ngrams[ngram_to_use] = {'count': 0, 'field_count': np.zeros(26, dtype=int)}
                        self.ngrams[ngram_to_use]['count'] += 1
                        if field_index != -1:
                            self.ngrams[ngram_to_use]['field_count'][field_index] += 1
            except Exception as e:
                print(f"Error processing row in ShortFourgramProcessor: {e}")

def run_processor(processor_class):
    processor = processor_class(is_local=True)
    output_file = processor.process_files()
    gc.collect()

    # Create filtered version
    filtered_output = os.path.join(os.path.dirname(output_file), f"filtered_{os.path.basename(output_file)}")
    filter_ngrams(output_file, filtered_output, min_count=5, min_zero_fields=1)
    gc.collect()

    # Small highly filtered version
    small_filtered_output = os.path.join(os.path.dirname(output_file), f"filtered_small_{os.path.basename(output_file)}")
    filter_ngrams(output_file, small_filtered_output, min_count=50, min_zero_fields=20)
    gc.collect()

    # Medium highly filtered version
    medium_filtered_output = os.path.join(os.path.dirname(output_file), f"filtered_medium_{os.path.basename(output_file)}")
    filter_ngrams(output_file, medium_filtered_output, min_count=50, min_zero_fields=15)
    gc.collect()

    # Create highly filtered version
    highly_filtered_output = os.path.join(os.path.dirname(output_file), f"filtered_high_{os.path.basename(output_file)}")
    filter_ngrams(output_file, highly_filtered_output, min_count=20, min_zero_fields=5)
    gc.collect()

    # Create highly filtered version
    very_highly_filtered_output = os.path.join(os.path.dirname(output_file), f"filtered_veryhigh_{os.path.basename(output_file)}")
    filter_ngrams(output_file, very_highly_filtered_output, min_count=20, min_zero_fields=3)
    gc.collect()

    # Create highly filtered version
    uber_highly_filtered_output = os.path.join(os.path.dirname(output_file), f"filtered_uberhigh_{os.path.basename(output_file)}")
    filter_ngrams(output_file, uber_highly_filtered_output, min_count=10, min_zero_fields=1)
    gc.collect()

    # Create filtered_three_subfield version
    filtered_three_subfield_output = os.path.join(os.path.dirname(output_file), f"filtered_three_subfield_{os.path.basename(output_file)}")
    filter_three_subfields(filtered_output, filtered_three_subfield_output, processor.subfield_int_map['id2label'])
    gc.collect()


if __name__ == "__main__":
    # NOTE: We removed shortFourgramProcessor because of memory management concerns. We could use this for 2TB of ram
    #     But certainly not for 1TB ram.

    initial_processors = [
        FullUnigramProcessor,
        ShortUnigramProcessor,
        FullBigramProcessor,
        ShortBigramProcessor,
        FullTrigramProcessor,
        ShortTrigramProcessor,
        FullFourgramProcessor,
        # ShortFourgramProcessor
    ]

    with mp.Pool(processes=min(mp.cpu_count(), len(initial_processors))) as pool:
        pool.map(run_processor, initial_processors)

    print("Initial processors completed successfully.")

    # Run post-processing
    post_processor = KeyPhraseConstructor(input_dir=r"E:\HugeDatasetBackup\ngram_mining_data",
                                   output_dir=os.path.join(r"E:\HugeDatasetBackup\ngram_mining_data", "data", "output"))
    post_processor.run()

    print("Post-processing completed successfully.")