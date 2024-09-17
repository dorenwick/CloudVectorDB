import gc
import json
import os
import time
from collections import Counter
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from span_marker import SpanMarkerModel
from tqdm import tqdm
import concurrent.futures
from pygtrie import CharTrie


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper


class EfficientCounter:
    def __init__(self):
        self.trie = CharTrie()

    def update(self, items):
        for item, count in items:
            self.trie[item] = self.trie.get(item, 0) + count

    def merge(self, other):
        for key, value in other.trie.items():
            self.trie[key] = self.trie.get(key, 0) + value

    def items(self):
        return self.trie.items()

    def __len__(self):
        return len(self.trie)


class AbstractDataConstructionMultiGPU:
    """
    python AbstractDataConstructionMultiGPU.py
    conda activate cite-grab

    """

    def __init__(self, input_dir: str,
                 output_dir: str,
                 keyphrase_model_path: str,
                 embedding_model_path: str,
                 batch_size: int = 100_000,
                 extract_keywords: bool = True,
                 generate_embeddings: bool = True):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.extract_keywords = extract_keywords
        self.generate_embeddings_bool = generate_embeddings
        self.ensure_output_directory()

        self.setup_gpu()
        self.initialize_models(keyphrase_model_path, embedding_model_path)
        self.field_int_map = self.load_or_create_field_int_map()

        self.full_unigrams = EfficientCounter()
        self.full_bigrams = EfficientCounter()
        self.short_unigrams = EfficientCounter()
        self.short_bigrams = EfficientCounter()
        self.batch_counters = []
        self.batches_before_merge = 10

    def setup_gpu(self):
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("CUDA device name:", torch.cuda.get_device_name(0))

        num_gpus = torch.cuda.device_count()
        print("num_gpus: ", num_gpus)
        if num_gpus > 0:
            self.devices = [f"cuda:{i}" for i in range(num_gpus)]
            print(f"Using {num_gpus} GPU(s): {self.devices}")
            self.keyphrase_device = torch.device(self.devices[0])
            self.embedding_device = self.devices[1] if len(self.devices) > 1 else self.devices[0]
        else:
            print("No GPUs detected. Using CPU.")
            self.devices = ["cpu"]
            self.keyphrase_device = torch.device("cpu")
            self.embedding_device = "cpu"
        print("self.keyphrase_device ", self.keyphrase_device)
        print("self.embedding_device ", self.embedding_device)

    def initialize_models(self, keyphrase_model_path, embedding_model_path):
        self.keyphrase_model = SpanMarkerModel.from_pretrained(keyphrase_model_path).to(self.keyphrase_device)
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.embedding_device)

    def load_or_create_field_int_map(self) -> Dict[str, Dict[str, int]]:
        field_int_map_path = os.path.join(self.output_dir, "field_int_map.json")
        if os.path.exists(field_int_map_path):
            with open(field_int_map_path, 'r') as f:
                return json.load(f)
        else:
            # Create field_int_map based on the provided mapping
            id2label = {
                0: "Economics, Econometrics and Finance",
                1: "Materials Science",
                2: "Environmental Science",
                3: "Medicine",
                4: "Psychology",
                5: "Dentistry",
                6: "Business, Management and Accounting",
                7: "Engineering",
                8: "Biochemistry, Genetics and Molecular Biology",
                9: "Agricultural and Biological Sciences",
                10: "Energy",
                11: "Earth and Planetary Sciences",
                12: "Health Professions",
                13: "Chemistry",
                14: "Chemical Engineering",
                15: "Social Sciences",
                16: "Pharmacology, Toxicology and Pharmaceutics",
                17: "Arts and Humanities",
                18: "Mathematics",
                19: "Immunology and Microbiology",
                20: "Veterinary",
                21: "Decision Sciences",
                22: "Nursing",
                23: "Physics and Astronomy",
                24: "Neuroscience",
                25: "Computer Science"
            }
            label2id = {v: k for k, v in id2label.items()}
            field_int_map = {"id2label": id2label, "label2id": label2id}
            with open(field_int_map_path, 'w') as f:
                json.dump(field_int_map, f)
            return field_int_map

    def extract_entities(self, texts: List[str], model: SpanMarkerModel) -> List[List[Dict]]:
        with torch.cuda.device(self.keyphrase_device):
            return model.predict(texts)

    @measure_time
    def generate_embeddings(self, texts: List[str], quantize_embeddings: bool = False) -> np.ndarray:
        with torch.cuda.device(self.embedding_device):
            if quantize_embeddings:
                embeddings = self.embedding_model.encode(texts, batch_size=256, convert_to_tensor=True, precision="binary",
                                                         show_progress_bar=True)
            else:
                embeddings = self.embedding_model.encode(texts, batch_size=256, convert_to_tensor=True, show_progress_bar=True)

            # Convert to numpy and ensure it's a 2D array
            embeddings_np = embeddings.cpu().numpy()

            if embeddings_np.ndim == 1:
                embeddings_np = embeddings_np.reshape(1, -1)

            print(f"Embeddings shape: {embeddings_np.shape}")

            return embeddings_np


    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        print("batch: ", batch.head(1).to_string())

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            if self.extract_keywords:
                futures.append(executor.submit(self.extract_keywords_from_batch, batch))
            if self.generate_embeddings_bool:
                futures.append(executor.submit(self.generate_embeddings_for_batch, batch))

            for future in concurrent.futures.as_completed(futures):
                batch = future.result()

        if not self.extract_keywords:
            batch['keywords_title'] = [[] for _ in range(len(batch))]
            batch['keywords_abstract'] = [[] for _ in range(len(batch))]
            print("Keyword extraction skipped.")

        if not self.generate_embeddings_bool:
            print("Embedding generation skipped.")

        return batch

    def extract_keywords_from_batch(self, batch):
        try:
            batch['keywords_title'] = [[] for _ in range(len(batch))]
            batch['keywords_abstract'] = [[] for _ in range(len(batch))]

            non_empty_titles = [(i, title) for i, title in enumerate(batch['title']) if
                                isinstance(title, str) and title.strip()]
            if non_empty_titles:
                indices, titles = zip(*non_empty_titles)
                title_keywords = self.extract_entities(titles, self.keyphrase_model)
                for i, keywords in zip(indices, title_keywords):
                    batch.at[batch.index[i], 'keywords_title'] = keywords

            non_empty_abstracts = [(i, abstract) for i, abstract in enumerate(batch['abstract_string']) if
                                   isinstance(abstract, str) and abstract.strip()]
            if non_empty_abstracts:
                indices, abstracts = zip(*non_empty_abstracts)
                abstract_keywords = self.extract_entities(abstracts, self.keyphrase_model)
                for i, keywords in zip(indices, abstract_keywords):
                    batch.at[batch.index[i], 'keywords_abstract'] = keywords

            print(
                f"Processed {len(non_empty_titles)} non-empty titles and {len(non_empty_abstracts)} non-empty abstracts.")
        except Exception as e:
            print(f"Error in extract_keywords: {str(e)}")
            print(f"Sample title: {batch['title'].iloc[0] if len(batch) > 0 else 'No titles'}")
            print(f"Sample abstract: {batch['abstract_string'].iloc[0] if len(batch) > 0 else 'No abstracts'}")

        return batch

    def generate_embeddings_for_batch(self, batch):
        try:
            batch['full_string'] = batch.apply(lambda row:
                                               f"{row['title']} {row['authors_string']} {row['field']} {row['subfield']} {row['topic']} " +
                                               f"{' '.join([k['span'] for k in row.get('keywords_title', []) + row.get('keywords_abstract', [])])}".strip(),
                                               axis=1)
            batch['topic_string'] = batch.apply(lambda row:
                                                f"{row['title']} {row['field']} {row['subfield']} {row['topic']} " +
                                                f"{' '.join([k['span'] for k in row.get('keywords_title', []) + row.get('keywords_abstract', [])])}".strip(),
                                                axis=1)

            batch['full_string_embeddings'] = list(self.generate_embeddings(batch['full_string'].tolist()))
            batch['abstract_string_embeddings'] = list(self.generate_embeddings(batch['abstract_string'].tolist()))
            batch['topic_string_embeddings'] = list(self.generate_embeddings(batch['topic_string'].tolist()))
            batch['topic_string_embeddings_binary'] = list(
                self.generate_embeddings(batch['topic_string'].tolist(), quantize_embeddings=True))

        except Exception as e:
            print(f"Error in generate_embeddings: {str(e)}")

        return batch

    def process_batch_ngrams(self, df):
        full_text = df.apply(lambda row: f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower(),
                             axis=1)
        short_text = df.apply(lambda row: f"{row['title']} {row['authors_string']}".lower(), axis=1)

        batch_full_unigrams = Counter(word for text in full_text for word in text.split())
        batch_short_unigrams = Counter(word for text in short_text for word in text.split())
        batch_full_bigrams = Counter(
            ' '.join(pair) for text in full_text for pair in zip(text.split()[:-1], text.split()[1:]))
        batch_short_bigrams = Counter(
            ' '.join(pair) for text in short_text for pair in zip(text.split()[:-1], text.split()[1:]))

        return (batch_full_unigrams, batch_full_bigrams, batch_short_unigrams, batch_short_bigrams)

    @measure_time
    def merge_batch_counters(self):
        for batch_counters in self.batch_counters:
            self.short_unigrams.update(batch_counters[2].items())
            self.short_bigrams.update(batch_counters[3].items())
            self.full_unigrams.update(batch_counters[0].items())
            self.full_bigrams.update(batch_counters[1].items())
        self.batch_counters.clear()

    @measure_time
    def update_ngram_counters(self, df: pd.DataFrame):
        batch_counters = self.process_batch_ngrams(df)
        self.batch_counters.append(batch_counters)

        if len(self.batch_counters) >= self.batches_before_merge:
            self.merge_batch_counters()

    def save_ngram_data(self):
        def save_counter(counter: EfficientCounter, file_name: str):
            df = pd.DataFrame(counter.items(), columns=['ngram', 'count'])
            df['smoothed_score'] = 0.0  # Default value
            df['ctf_idf_score'] = 0.0  # Default value
            df['field_count'] = [np.zeros(26, dtype=int) for _ in range(len(df))]  # Placeholder for field counts
            df.to_parquet(os.path.join(self.output_dir, file_name), index=False)

        save_counter(self.full_unigrams, "full_string_unigrams.parquet")
        save_counter(self.full_bigrams, "full_string_bigrams.parquet")
        save_counter(self.short_unigrams, "short_unigrams.parquet")
        save_counter(self.short_bigrams, "short_bigrams.parquet")

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])
        counter = 0

        for file_name in tqdm(input_files, desc="Processing files"):
            try:
                input_path = os.path.join(self.input_dir, file_name)
                output_path = os.path.join(self.output_dir, f"processed_{file_name}")

                if os.path.exists(output_path):
                    print(f"Skipping {file_name} as it has already been processed.")
                    continue

                df = pd.read_parquet(input_path)
                processed_df = self.process_batch(df)
                self.update_ngram_counters(processed_df)
                self.save_processed_batch(processed_df, output_path)

                print(f"Processed {file_name}")
                print(processed_df.head(1).to_string())
                print(f"Current n-gram counter lengths:")
                print(f"Full unigrams: {len(self.full_unigrams)}")
                print(f"Full bigrams: {len(self.full_bigrams)}")
                print(f"Short unigrams: {len(self.short_unigrams)}")
                print(f"Short bigrams: {len(self.short_bigrams)}")

                counter += 1
                if counter % 100 == 0:
                    self.save_ngram_data()

                gc.collect()
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue  # Skip to the next file if there's an error

        self.merge_batch_counters()
        self.save_ngram_data()
        print("All files processed successfully.")

    def ensure_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    @measure_time
    def save_processed_batch(self, df: pd.DataFrame, output_path: str):
        columns_to_save = [
            'work_id', 'works_int_id', 'has_abstract', 'title', 'authors_string', 'abstract_string',
            'field', 'subfield', 'topic', 'keywords_title', 'keywords_abstract', 'full_string',
            'topic_string', 'full_string_embeddings', 'abstract_string_embeddings',
            'topic_string_embeddings', 'topic_string_embeddings_binary'
        ]

        columns_to_save = [col for col in columns_to_save if col in df.columns]
        print(f"Columns being saved: {columns_to_save}")
        df[columns_to_save].to_parquet(output_path, index=False)

    def clean_ngrams(self, df: pd.DataFrame) -> pd.DataFrame:
        def is_valid_ngram(ngram: str) -> bool:
            non_alpha_count = sum(1 for char in ngram if not char.isalpha() and char not in ["'", '"', '.', '$'])
            return non_alpha_count < 2

        df_cleaned = df[(df['count'] > 1) | ((df['count'] == 1) & (df['ngram'].apply(is_valid_ngram)))]
        return df_cleaned

    def calculate_non_zero_counts(self, df: pd.DataFrame):
        df['non_zero_count'] = df['field_count'].apply(lambda x: np.count_nonzero(x))
        return df

    def calculate_ctf_idf_score(self, df: pd.DataFrame):
        B = 26  # Number of fields
        df['ctf_idf_score'] = (B / df['non_zero_count']) / np.log1p(df['count'])
        return df


    def post_process_ngram_data(self):
        for file_name in ['full_string_unigrams.parquet', 'full_string_bigrams.parquet',
                          'short_unigrams.parquet', 'short_bigrams.parquet']:
            file_path = os.path.join(self.output_dir, file_name)
            df = pd.read_parquet(file_path)

            print(f"Processing {file_name}")
            print(f"Total rows before cleaning: {len(df)}")

            df_cleaned = self.clean_ngrams(df)
            print(f"Total rows after cleaning: {len(df_cleaned)}")

            df_cleaned = self.calculate_non_zero_counts(df_cleaned)
            df_cleaned = self.calculate_ctf_idf_score(df_cleaned)
            df_cleaned.to_parquet(file_path, index=False)

    def run(self):
        self.process_files()
        print("Files processed.")
        self.post_process_ngram_data()
        print("Data processing completed successfully.")

if __name__ == "__main__":
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    keyphrase_model_path = "/workspace/models/models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec/snapshots/bfc31646972e22ebf331c2e877c30439f01d35b3"
    embedding_model_path = "/workspace/models/models--Snowflake--snowflake-arctic-embed-xs/snapshots/86a07656cc240af5c7fd07bac2f05baaafd60401"

    processor = AbstractDataConstructionMultiGPU(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path,
        embedding_model_path=embedding_model_path,
        extract_keywords=False,  # Set this to False to skip keyword extraction
        generate_embeddings=True  # Set this to False to skip embedding generation
    )
    processor.run()