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

    python AbstractDataConstructionMultiGPU.py

    CloudVectorDB

    A script for running on very powerful cloud computer, for building a very large dataset of triplets, then training encoders, then building the embeddings with the encoder, then building the vectordb with the encoder.

    Installation Instructions for Ubuntu Server

    Update and Upgrade System sudo apt update && sudo apt upgrade -y
    Install CUDA Follow the official NVIDIA instructions to install CUDA on your Ubuntu server. The exact steps may vary depending on your Ubuntu version and desired CUDA version. Generally, it involves:
    Verify you have a CUDA-capable GPU Download and install the NVIDIA CUDA Toolkit Set up the required environment variables

    Install Miniconda wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh bash Miniconda3-latest-Linux-x86_64.sh Follow the prompts to install Miniconda. After installation, close and reopen your terminal.
    Create and Activate Conda Environment conda create -n cite_grabber python=3.10 conda activate cite_grabber
    Install PyTorch 2.4.0 conda install pytorch==2.4.0 torchvision torchaudio cudatoolkit=11.8 -c pytorch
    Install Transformers 4.39.0 pip install transformers==4.39.0
    Install Sentence-Transformers 3.0.1 pip install sentence-transformers==3.0.1
    Install Additional Required Packages pip install pandas numpy tqdm pyarrow span-marker

    conda install pandas numpy tqdm pyarrow span-marker

    Set Up Project Directory mkdir -p /workspace/data/input mkdir -p /workspace/data/output mkdir -p /workspace/models
    Download Required Models Download the necessary models (keyphrase model and embedding model) and place them in the /workspace/models directory.

    Run the Script python AbstractDataConstructionMultiGPU.py  Remember to adjust these instructions if you have specific requirements or if your setup differs from a standard Ubuntu server environment.

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


    TODO: We want to clean ngrams in the following way:
        We wish to go through unigrams and bigrams afterwards and do this: For any unigram or bigram that has count=1 and 2 or more characters that are not alphabetical or in an apostrophe,
         speechmark, or fullstop, or dollar signs
            we shall remove the row from the parquet table/dataframe.
            So, make a method that does this. I also want you to print the total number of rows before and after we do the cleaning.



    TODO: We shall be running this on 2xRTX 4090, so make sure we use both gpu's.

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

        import torch

        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("CUDA device name:", torch.cuda.get_device_name(0))

        # Set up multi-GPU or fall back to CPU
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

        # Initialize models using provided paths
        self.keyphrase_model = SpanMarkerModel.from_pretrained(keyphrase_model_path).to(self.keyphrase_device)
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.embedding_device)

        # Rest of the initialization code remains the same
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
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, precision="binary",
                                                         show_progress_bar=True)
            else:
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings.cpu().numpy()

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        def extract_keywords():
            try:
                batch['keywords_title'] = self.extract_entities(batch['title'].tolist(), self.keyphrase_model)
                batch['keywords_abstract'] = self.extract_entities(batch['abstract_string'].tolist(),
                                                                   self.keyphrase_model)
            except Exception as e:
                print(f"Error in extract_keywords: {str(e)}")
                # Initialize empty lists if extraction fails
                batch['keywords_title'] = [[] for _ in range(len(batch))]
                batch['keywords_abstract'] = [[] for _ in range(len(batch))]

        def generate_embeddings():
            try:
                batch['full_string'] = batch.apply(lambda row:
                                                   f"{row['title']} {row['authors_string']} {row['field']} {row['subfield']} {row['topic']} " +
                                                   f"{' '.join([k['span'] for k in row.get('keywords_title', []) + row.get('keywords_abstract', [])])}".strip(),
                                                   axis=1)
                batch['topic_string'] = batch.apply(lambda row:
                                                    f"{row['title']} {row['field']} {row['subfield']} {row['topic']} " +
                                                    f"{' '.join([k['span'] for k in row.get('keywords_title', []) + row.get('keywords_abstract', [])])}".strip(),
                                                    axis=1)

                batch['full_string_embeddings'] = self.generate_embeddings(batch['full_string'].tolist())
                batch['abstract_string_embeddings'] = self.generate_embeddings(batch['abstract_string'].tolist())
                batch['abstract_string_embeddings_binary'] = self.generate_embeddings(batch['abstract_string'].tolist(),
                                                                                      quantize_embeddings=True)
                batch['topic_string_embeddings'] = self.generate_embeddings(batch['topic_string'].tolist())
                batch['topic_string_embeddings_binary'] = self.generate_embeddings(batch['topic_string'].tolist(),
                                                                                   quantize_embeddings=True)
            except Exception as e:
                print(f"Error in generate_embeddings: {str(e)}")

        # Use ThreadPoolExecutor to run tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            keyword_future = executor.submit(extract_keywords)
            embedding_future = executor.submit(generate_embeddings)

            # Wait for both tasks to complete
            concurrent.futures.wait([keyword_future, embedding_future])

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

                # Save keyword data
                self.save_entity_data(processed_df, 'keywords')

                # Print progress information
                print(f"Processed {file_name}")
                print(processed_df.head(2).to_string())
                print(f"Current n-gram counter lengths:")
                print(f"Full unigrams: {len(self.full_unigrams)}")
                print(f"Full bigrams: {len(self.full_bigrams)}")
                print(f"Short unigrams: {len(self.short_unigrams)}")
                print(f"Short bigrams: {len(self.short_bigrams)}")
                gc.collect()
            except Exception as e:
                print(f"Error: {e}")
        self.save_ngram_data()
        print("All files processed successfully.")

    @measure_time
    def save_processed_batch(self, df: pd.DataFrame, output_path: str):
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

        # Only save columns that exist in the DataFrame
        columns_to_save = [col for col in columns_to_save if col in df.columns]

        print(f"Columns being saved: {columns_to_save}")

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

    def clean_ngrams(self, df: pd.DataFrame) -> pd.DataFrame:
        def is_valid_ngram(ngram: str) -> bool:
            # Check if the ngram has 2 or more non-alphabetic characters
            # (excluding apostrophes, speech marks, full stops, and dollar signs)
            non_alpha_count = sum(1 for char in ngram if not char.isalpha() and char not in ["'", '"', '.', '$'])
            return non_alpha_count < 2

        # Filter out rows with count=1 and invalid ngrams
        df_cleaned = df[(df['count'] > 1) | ((df['count'] == 1) & (df['ngram'].apply(is_valid_ngram)))]

        return df_cleaned

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
        embedding_model_path=embedding_model_path
    )
    processor.run()