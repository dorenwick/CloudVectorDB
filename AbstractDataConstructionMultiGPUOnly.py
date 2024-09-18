import gc
import json
import os
import re
import time
from collections import Counter, defaultdict
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


class AbstractDataConstructionMultiGPUOnly():
    """

    python BothProgramsAtOnce.py

    python AbstractDataConstructionMultiGPUOnly &
    python AbstractEmbeddingGenerator.py

    python AbstractDataConstructionMultiGPUOnly.py

    python AbstractDataConstructionMultiGPUOnly.py

    conda activate cite_grabber

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



    TODO: We shall be running this on 2xRTX 4090, so make sure we use both gpu's.

    Then, we shall

    TODO: We shall run keywords extraction on 1gpu.


    """

    def __init__(self, input_dir: str,
                 output_dir: str,
                 keyphrase_model_path: str,
                 embedding_model_path: str,
                 batch_size: int = 100_000,
                 extract_keywords: bool = True,
                 generate_embeddings: bool = True):  # New parameter

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        self.extract_keywords = extract_keywords
        self.generate_embeddings_bool = generate_embeddings

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

        self.full_unigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.full_bigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.short_unigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})
        self.short_bigrams = defaultdict(lambda: {'count': 0, 'field_count': np.zeros(26, dtype=int)})

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
                embeddings = self.embedding_model.encode(texts, batch_size=512, convert_to_tensor=True,
                                                         precision="binary",
                                                         show_progress_bar=True)
            else:
                embeddings = self.embedding_model.encode(texts, batch_size=512, convert_to_tensor=True,
                                                         show_progress_bar=True)

            # Convert to numpy and ensure it's a 2D array
            embeddings_np = embeddings.cpu().numpy()

            if embeddings_np.ndim == 1:
                embeddings_np = embeddings_np.reshape(1, -1)

            print(f"Embeddings shape: {embeddings_np.shape}")

            return embeddings_np

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:

        def extract_keywords():
            if not self.extract_keywords:
                batch['keywords_title'] = [[] for _ in range(len(batch))]
                batch['keywords_abstract'] = [[] for _ in range(len(batch))]
                print("Keyword extraction skipped.")
                return

            try:
                # Initialize with empty lists for all rows
                batch['keywords_title'] = [[] for _ in range(len(batch))]
                batch['keywords_abstract'] = [[] for _ in range(len(batch))]

                # Limit to first 200 rows for keyphrase extraction
                limited_batch = batch.head(500)

                # Process non-empty titles
                non_empty_titles = [title for title in limited_batch['title'] if
                                    isinstance(title, str) and title.strip()]
                if non_empty_titles:
                    title_keywords = self.extract_entities(non_empty_titles, self.keyphrase_model)
                    for title, keywords in zip(non_empty_titles, title_keywords):
                        idx = batch.index[batch['title'] == title].tolist()
                        if idx:
                            batch.at[idx[0], 'keywords_title'] = keywords

                # Process non-empty abstracts
                non_empty_abstracts = [abstract for abstract in limited_batch['abstract_string'] if
                                       isinstance(abstract, str) and abstract.strip()]
                if non_empty_abstracts:
                    abstract_keywords = self.extract_entities(non_empty_abstracts, self.keyphrase_model)
                    for abstract, keywords in zip(non_empty_abstracts, abstract_keywords):
                        idx = batch.index[batch['abstract_string'] == abstract].tolist()
                        if idx:
                            batch.at[idx[0], 'keywords_abstract'] = keywords

                print(
                    f"Processed {len(non_empty_titles)} non-empty titles and {len(non_empty_abstracts)} non-empty abstracts (limited to first 500).")
            except Exception as e:
                print(f"Error in extract_keywords: {str(e)}")
                print(f"Sample title: {batch['title'].iloc[0] if len(batch) > 0 else 'No titles'}")
                print(f"Sample abstract: {batch['abstract_string'].iloc[0] if len(batch) > 0 else 'No abstracts'}")

        def generate_embeddings():
            if not self.generate_embeddings_bool:
                print("Embedding generation skipped.")
                return

            try:
                # batch['full_string'] = batch.apply(lambda row:
                #                                    f"{row['title']} {row['authors_string']} {row['field']} {row['subfield']} {row['topic']} " +
                #                                    f"{' '.join([k['span'] for k in row.get('keywords_title', []) + row.get('keywords_abstract', [])])}".strip(),
                #                                    axis=1)
                # batch['topic_string'] = batch.apply(lambda row:
                #                                     f"{row['title']} {row['field']} {row['subfield']} {row['topic']} " +
                #                                     f"{' '.join([k['span'] for k in row.get('keywords_title', []) + row.get('keywords_abstract', [])])}".strip(),
                #                                     axis=1)
                #
                # full_string_embeddings = self.generate_embeddings(batch['full_string'].tolist())
                # print("full_string_embeddings done")
                #
                # topic_string_embeddings = self.generate_embeddings(batch['topic_string'].tolist())
                # topic_string_embeddings_binary = self.generate_embeddings(batch['topic_string'].tolist(),
                #                                                           quantize_embeddings=True)

                abstract_string_embeddings = self.generate_embeddings(batch['abstract_string'].tolist())

                # batch['full_string_embeddings'] = list(full_string_embeddings)
                batch['abstract_string_embeddings'] = list(abstract_string_embeddings)
                # batch['topic_string_embeddings'] = list(topic_string_embeddings)
                # batch['topic_string_embeddings_binary'] = list(topic_string_embeddings_binary)

            except Exception as e:
                print(f"Error in generate_embeddings: {str(e)}")

        # Use ThreadPoolExecutor to run tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            keyword_future = executor.submit(extract_keywords)
            embedding_future = executor.submit(generate_embeddings)

            # Wait for both tasks to complete
            concurrent.futures.wait([keyword_future, embedding_future])

        return batch

    def is_valid_ngram(self, ngram: str) -> bool:
        valid_chars = set("'\".$?<>:;,")
        non_alpha_count = sum(1 for char in ngram if not char.isalpha() and char not in valid_chars)
        return non_alpha_count < 2 and len(ngram) > 0

    def update_ngram_counters(self, df: pd.DataFrame):
        full_unigrams, short_unigrams, full_bigrams, short_bigrams = update_ngram_counters(df, self.field_int_map)

        # Update the class attributes
        self.full_unigrams.update(full_unigrams.to_dict('index'))
        self.short_unigrams.update(short_unigrams.to_dict('index'))
        self.full_bigrams.update(full_bigrams.to_dict('index'))
        self.short_bigrams.update(short_bigrams.to_dict('index'))

    def calculate_non_zero_counts(self, df: pd.DataFrame):
        df['non_zero_count'] = df['field_count'].apply(lambda x: np.count_nonzero(np.array(x)))
        return df

    def calculate_ctf_idf_score(self, df: pd.DataFrame):
        B = 26  # Number of fields
        df['ctf_idf_score'] = ((B / (df['non_zero_count'] + 1))) / np.log1p(df['count'])
        return df


    def calculate_threshold_based_ctf_idf_score(self, df: pd.DataFrame):
        B = 26  # Number of fields
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                      1.0]

        def calculate_score(row):
            total_count = row['count']
            field_counts = np.array(row['field_count'])

            # Calculate the number of occurrences needed to meet each threshold
            threshold_counts = [np.ceil(total_count * t) for t in thresholds]

            score = 1.0
            for i, count in enumerate(threshold_counts):
                # Count how many fields exceed this threshold
                fields_over_threshold = np.sum(field_counts >= count)

                # Apply boost for each field over the threshold
                boost = 1.0 + (thresholds[i] * 0.1)  # This makes the boost range from 1.1 to 2.0
                score *= boost ** (2 * fields_over_threshold)

            # Apply the original CTF-IDF formula as a base
            base_score = ((B / (row['non_zero_count'] + 1))) / np.log1p(total_count)

            return base_score * score

        df['threshold_ctf_idf_score'] = df.apply(calculate_score, axis=1)
        return df


    def save_ngram_data(self):
        def save_counter(counter, file_name: str):
            df = pd.DataFrame([
                {'ngram': k, 'count': v['count'], 'field_count': v['field_count'].tolist()}
                for k, v in counter.items()
            ])
            df.to_parquet(os.path.join(self.output_dir, file_name), index=False)

        save_counter(self.full_unigrams, "full_string_unigrams.parquet")
        save_counter(self.full_bigrams, "full_string_bigrams.parquet")
        save_counter(self.short_unigrams, "short_unigrams.parquet")
        save_counter(self.short_bigrams, "short_bigrams.parquet")

        print(f"N-gram data saved. Current counter sizes:")
        print(f"Full unigrams: {len(self.full_unigrams)}")
        print(f"Full bigrams: {len(self.full_bigrams)}")
        print(f"Short unigrams: {len(self.short_unigrams)}")
        print(f"Short bigrams: {len(self.short_bigrams)}")

    def post_process_ngram_data(self):
        for file_name in ['full_string_unigrams.parquet', 'full_string_bigrams.parquet',
                          'short_unigrams.parquet', 'short_bigrams.parquet']:
            file_path = os.path.join(self.output_dir, file_name)
            df = pd.read_parquet(file_path)

            print(f"Processing {file_name}")
            print(f"Total rows: {len(df)}")

            df = self.calculate_non_zero_counts(df)
            df = self.calculate_ctf_idf_score(df)
            df = self.calculate_threshold_based_ctf_idf_score(df)
            df.to_parquet(file_path, index=False)

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

                if self.extract_keywords or self.generate_embeddings_bool:
                    df = pd.read_parquet(input_path)
                    processed_df = self.process_batch(df)
                    self.update_ngram_counters(processed_df)
                    self.save_processed_batch(processed_df, output_path)
                else:
                    processed_df = pd.read_parquet(input_path)
                    self.update_ngram_counters(processed_df)

                if self.extract_keywords:
                    self.save_entity_data(processed_df, 'keywords')

                counter += 1
                if counter == 1 or counter % 200 == 0 or counter in [5, 10, 50]:
                    self.save_ngram_data()

                # Print progress information
                print(f"Processed {file_name}")

                gc.collect()
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        # Final save of n-gram data
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
                        'char_start_index': entity['char_start_index'],
                        'char_end_index': entity['char_end_index'],
                        'location': location
                    })

        entity_df = pd.DataFrame(entity_data)
        output_path = os.path.join(self.output_dir, f"{entity_type}_data.parquet")
        if os.path.exists(output_path):
            existing_df = pd.read_parquet(output_path)
            entity_df = pd.concat([existing_df, entity_df], ignore_index=True)
        entity_df.to_parquet(output_path, index=False)

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
        self.post_process_ngram_data()
        print("Data processing completed successfully.")


def update_ngram_counters(df: pd.DataFrame, field_int_map: dict):
    # Combine texts
    df['full_text'] = df['title'] + ' ' + df['authors_string'] + ' ' + df['abstract_string']
    df['short_text'] = df['title'] + ' ' + df['authors_string']

    # Convert field to numeric
    df['field_index'] = df['field'].map(field_int_map['label2id'])

    # Function to process unigrams
    def process_unigrams(text_series, field_index_series):
        # Split text into words
        words = text_series.str.lower().str.split()

        # Count unigrams
        unigram_counts = words.apply(Counter).sum()

        # Create field count matrix
        field_counts = pd.DataFrame({
            'word': words.explode(),
            'field_index': field_index_series.repeat(words.str.len())
        })
        field_counts = field_counts.groupby(['word', 'field_index']).size().unstack(fill_value=0)

        # Combine counts and field counts
        unigrams = pd.DataFrame({
            'count': unigram_counts,
            'field_count': field_counts.values.tolist()
        })

        return unigrams

    # Process full and short unigrams
    full_unigrams = process_unigrams(df['full_text'], df['field_index'])
    short_unigrams = process_unigrams(df['short_text'], df['field_index'])

    # Function to process bigrams
    def process_bigrams(text_series, field_index_series):
        # Generate bigrams
        words = text_series.str.lower().str.split()
        bigrams = words.apply(lambda x: [f"{x[i]} {x[i + 1]}" for i in range(len(x) - 1)])

        # Filter valid bigrams
        valid_bigrams = bigrams.apply(lambda x: [b for b in x if is_valid_ngram(b)])

        # Count bigrams
        bigram_counts = valid_bigrams.apply(Counter).sum()

        # Create field count matrix
        field_counts = pd.DataFrame({
            'bigram': valid_bigrams.explode(),
            'field_index': field_index_series.repeat(valid_bigrams.str.len())
        })
        field_counts = field_counts.groupby(['bigram', 'field_index']).size().unstack(fill_value=0)

        # Combine counts and field counts
        bigrams = pd.DataFrame({
            'count': bigram_counts,
            'field_count': field_counts.values.tolist()
        })

        return bigrams

    # Process full and short bigrams
    full_bigrams = process_bigrams(df['full_text'], df['field_index'])
    short_bigrams = process_bigrams(df['short_text'], df['field_index'])

    return full_unigrams, short_unigrams, full_bigrams, short_bigrams


# Update the is_valid_ngram function to work with vectorized operations
def is_valid_ngram(ngram: str) -> bool:
    valid_chars = set("'\".$?<>:;,")
    non_alpha_count = sum(1 for char in ngram if not char.isalpha() and char not in valid_chars)
    return non_alpha_count < 2 and len(ngram) > 0

if __name__ == "__main__":
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    keyphrase_model_path = "/workspace/models/models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec/snapshots/bfc31646972e22ebf331c2e877c30439f01d35b3"
    embedding_model_path = "/workspace/models/models--Snowflake--snowflake-arctic-embed-xs/snapshots/86a07656cc240af5c7fd07bac2f05baaafd60401"

    processor = AbstractDataConstructionMultiGPUOnly(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path,
        embedding_model_path=embedding_model_path,
        extract_keywords=False,  # Set this to False to skip keyword extraction
        generate_embeddings=False  # Set this to False to skip embedding generation
    )
    processor.run()