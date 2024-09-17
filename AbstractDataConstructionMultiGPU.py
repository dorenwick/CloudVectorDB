import os
import json
import time

import torch
import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from span_marker import SpanMarkerModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper

class AbstractDataConstructionMultiGPU():
    """
    TODO: We shall implement pathing somewhere, for our linux server.
        we shall make a directory called /workspace, and then copy the files from google drive (located in abstract_data)
        to /workspace

        within /workspace we shall have subdirectories called /models
        that we will put the models:

        keyphrase_model_path: str,
        embedding_model_path: str,

        models--Snowflake--snowflake-arctic-embed-xs\snapshots\86a07656cc240af5c7fd07bac2f05baaafd60401
        models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec\snapshots\bfc31646972e22ebf331c2e877c30439f01d35b3
        will be placed into

        /models

        These subdirectories are currently in abstract_data, keep that in mind.

        into.

    Then, we shall


    """


    def __init__(self, input_dir: str,
                 output_dir: str,
                 keyphrase_model_path: str,
                 embedding_model_path: str,
                 batch_size: int = 100_000):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models using provided paths
        self.keyphrase_model = SpanMarkerModel.from_pretrained(keyphrase_model_path).to(self.device)
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)

        # Load or create field_int_map
        self.field_int_map = self.load_or_create_field_int_map()
        print("self.field_int_map: ", self.field_int_map)

        # Initialize counters for n-grams
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
        return model.predict(texts)

    @measure_time
    def generate_embeddings(self, texts: List[str], quantize_embeddings: bool = False) -> np.ndarray:
        if quantize_embeddings:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, precision="binary",
                                                     show_progress_bar=True)
        else:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings.cpu().numpy()

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Extract keywords
        batch['keywords_title'] = self.extract_entities(batch['title'].tolist(), self.keyphrase_model)
        batch['keywords_abstract'] = self.extract_entities(batch['abstract_string'].tolist(), self.keyphrase_model)

        # Create full_string and topic_string
        batch['full_string'] = batch.apply(lambda row:
                                           f"{row['title']} {row['authors_string']} {row['field']} {row['subfield']} {row['topic']} {' '.join([k['span'] for k in row['keywords_title'] + row['keywords_abstract']])}".strip(),
                                           axis=1)
        batch['topic_string'] = batch.apply(lambda row:
                                            f"{row['title']} {row['field']} {row['subfield']} {row['topic']} {' '.join([k['span'] for k in row['keywords_title'] + row['keywords_abstract']])}".strip(),
                                            axis=1)

        # Generate embeddings
        batch['full_string_embeddings'] = self.generate_embeddings(batch['full_string'].tolist())
        batch['abstract_string_embeddings'] = self.generate_embeddings(batch['abstract_string'].tolist())
        batch['abstract_string_embeddings_binary'] = self.generate_embeddings(batch['abstract_string'].tolist(),
                                                                              quantize_embeddings=True)
        batch['topic_string_embeddings'] = self.generate_embeddings(batch['topic_string'].tolist())
        batch['topic_string_embeddings_binary'] = self.generate_embeddings(batch['topic_string'].tolist(),
                                                                           quantize_embeddings=True)

        return batch

    def update_ngram_counters(self, df: pd.DataFrame):
        full_text = df.apply(lambda row: f"{row['title']} {row['authors_string']} {row['abstract_string']}".lower(),
                             axis=1)
        short_text = df.apply(lambda row: f"{row['title']} {row['authors_string']}".lower(), axis=1)

        # Update unigrams
        self.full_unigrams.update(word for text in full_text for word in text.split())
        self.short_unigrams.update(word for text in short_text for word in text.split())

        # Update bigrams
        self.full_bigrams.update(
            ' '.join(pair) for text in full_text for pair in zip(text.split()[:-1], text.split()[1:]))
        self.short_bigrams.update(
            ' '.join(pair) for text in short_text for pair in zip(text.split()[:-1], text.split()[1:]))

    def save_ngram_data(self):
        def save_counter(counter: Counter, file_name: str):
            df = pd.DataFrame([(k, v) for k, v in counter.items()], columns=['ngram', 'count'])
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

        for file_name in tqdm(input_files, desc="Processing files"):
            input_path = os.path.join(self.input_dir, file_name)
            output_path = os.path.join(self.output_dir, f"processed_{file_name}")

            if os.path.exists(output_path):
                print(f"Skipping {file_name} as it has already been processed.")
                continue

            df = pd.read_parquet(input_path)
            processed_df = self.process_batch(df)
            self.update_ngram_counters(processed_df)
            self.save_processed_batch(processed_df, output_path)

            # Save keyword data
            self.save_entity_data(processed_df, 'keywords')

        self.save_ngram_data()
        print("All files processed successfully.")

    def save_processed_batch(self, df: pd.DataFrame, output_path: str):
        # Select only the columns we want to save
        columns_to_save = [
            'work_id',
            'works_int_id',
            'has_abstract',
            'title',
            'authors_string',
            'abstract_string',
            'field',
            'subfield',
            'topic',
            'keywords_title',
            'keywords_abstract',
            'full_string',
            'topic_string',
            'full_string_embeddings',
            'abstract_string_embeddings',
            'abstract_string_embeddings_binary',
            'topic_string_embeddings',
            'topic_string_embeddings_binary'
        ]
        df[columns_to_save].to_parquet(output_path, index=False)

    def save_entity_data(self, df: pd.DataFrame, entity_type: str):
        entity_data = []
        for _, row in df.iterrows():
            work_id = row['work_id']
            for location in ['title', 'abstract']:
                entities = row[f'{entity_type}_{location}']
                for entity in entities:
                    entity_data.append({
                        'work_id': work_id,
                        'entity': entity['span'],
                        'score': entity['score'],
                        'char_start': entity['char_start'],
                        'char_end': entity['char_end'],
                        'location': location
                    })

        entity_df = pd.DataFrame(entity_data)
        output_path = os.path.join(self.output_dir, f"{entity_type}_data.parquet")
        if os.path.exists(output_path):
            existing_df = pd.read_parquet(output_path)
            entity_df = pd.concat([existing_df, entity_df], ignore_index=True)
        entity_df.to_parquet(output_path, index=False)

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
            df = self.calculate_non_zero_counts(df)
            df = self.calculate_ctf_idf_score(df)
            df.to_parquet(file_path, index=False)

    def run(self):
        self.process_files()
        self.post_process_ngram_data()
        print("Data processing completed successfully.")

if __name__ == "__main__":



    input_dir = "/path/to/input/directory"
    output_dir = "/path/to/output/directory"
    keyphrase_model_path = "/path/to/keyphrase/model"
    embedding_model_path = "/path/to/embedding/model"

    processor = AbstractDataConstructionMultiGPU(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path,
        embedding_model_path=embedding_model_path
    )
    processor.run()