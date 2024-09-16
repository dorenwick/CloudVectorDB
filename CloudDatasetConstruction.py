import gc
import os
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
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



class CloudDatasetConstruction:
    """




    TODO: We wish to make this class run on a linux server.
    TODO: We will have to put mongodb collections there first.

    TODO: We want to add a system for building unigrams and bigrams from:
        abstract_string (specifically abstract string only).
        we will do unigrams and bigrams.
        We shall also do run the keyword classifier on abstracts to get trigrams and 4+grams.
        We could use all gpu's for the keyword classifier, (distributed computing).




    TODO: So, make a abstract_unigrams.parquet file and abstract_bigrams.parquet file
      and abstract_keyphrases.parquet file where

      abstract_unigrams.parquet has columns (unigram_type, unigram_count, unigram_score)
      abstract_bigrams.parquet has columns (bigram_type, bigram_count, bigram_score)

      abstract_keyphrases.parquet has columns (keyphrase, work_id, score, field, char_start, char_end)

      This abstract_keyphrases.parquet file will be recording literally every single occurance of keyphrase entity detected, and
      keyphrase shall not be unique (it will be the entity_text or span_text of the keyphrase. We will record the char_start and char_end of all this as well.

      abstract_data.parquet will have columns (abstract_string, field, subfield, topic, keywords, work_id)

    So, we will build all three of these parquet files simultaneously, and save them all once done.
    the keywords in abstract_data.parquet will be set up empty here, and filled in later in a post-processing phase.







    """

    def __init__(self):
        self.model_path = r"C:\Users\doren\.cache\huggingface\hub\models--Snowflake--snowflake-arctic-embed-xs\snapshots\86a07656cc240af5c7fd07bac2f05baaafd60401"

        self.mongo_url = "mongodb://localhost:27017/"
        self.mongo_database_name = "OpenAlex"
        self.mongo_works_collection_name = "Works"
        self.output_directory = "E:\\HugeDatasetBackup\\DATA_CITATION_GRABBER\\datasets_collected"


    def collect_abstracts(self, batch_size=1_000_000, max_works=None):
        client = MongoClient(self.mongo_url)
        db = client[self.mongo_database_name]
        works_collection = db[self.mongo_works_collection_name]

        output_file = os.path.join(self.output_directory, "works_abstracts.parquet")

        # Initialize an empty list to store dictionaries
        data = []

        # Initialize counters
        total_processed = 0
        total_with_abstract = 0
        batch_counter = 0

        # Use tqdm for progress tracking
        for work in tqdm(works_collection.find({"abstract_inverted_index": {"$exists": True}}),
                         desc="Processing works", unit="work"):
            work_id = work.get('id')
            abstract = self.reconstruct_abstract(work.get('abstract_inverted_index', {}))

            if abstract:
                primary_topic = work.get('primary_topic', {})
                if primary_topic:
                    field = primary_topic.get('field', {}).get('display_name', '')
                    subfield = primary_topic.get('subfield', {}).get('display_name', '')
                    topic = primary_topic.get('display_name', '')
                else:
                    field = ''
                    subfield = ''
                    topic = ''

                data.append({
                    'work_id': work_id,
                    'abstract_string': abstract,
                    'field': field,
                    'subfield': subfield,
                    'topic': topic
                })

                total_with_abstract += 1

            total_processed += 1

            # Save batch to parquet file
            if len(data) >= batch_size:
                self.save_batch(data, batch_counter)
                data = []  # Clear the list after saving
                batch_counter += 1

            if max_works and total_processed >= max_works:
                break

        # Save any remaining data
        if data:
            self.save_batch(data, batch_counter)

        client.close()

        print(f"Total works processed: {total_processed}")
        print(f"Works with abstracts: {total_with_abstract}")
        print(f"Parquet file saved to: {output_file}")

    def save_batch(self, results, batch_counter):
        df = pd.DataFrame(results, columns=[
            'work_id', 'abstract_string', 'field', 'subfield', 'topic'
        ])

        print(f"\nFirst 10 rows of batch {batch_counter}:")
        print(df.head(10).to_string())

        print(f"\nLast 10 rows of batch {batch_counter}:")
        print(df.tail(10).to_string())

        output_file = os.path.join(self.output_directory, f"works_abstracts_batch_{batch_counter}.parquet")
        df.to_parquet(output_file, index=False)
        print(f"Saved batch {batch_counter} to {output_file}")

    def load_abstracts(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.output_directory, "works_abstracts.parquet")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} abstracts from {file_path}")
        return df


    def reconstruct_abstract(self, inverted_index):
        if not inverted_index:
            return ""

        # Flatten the inverted index
        flat_abstract = []
        for word, positions in inverted_index.items():
            for position in positions:
                flat_abstract.append((position, word))

        # Sort by position and join words
        return " ".join(word for _, word in sorted(flat_abstract))

    def process_abstracts_with_embeddings(self, batch_size=10_000, quantize_embeddings=False):
        # Load the encoding model
        model = SentenceTransformer(self.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Get all batch files
        all_files = os.listdir(self.output_directory)
        batch_files = sorted([f for f in all_files
                              if f.startswith("works_abstracts_batch_") and f.endswith(".parquet") and not f.endswith(
                "_encoded.parquet")],
                             key=lambda x: int(re.search(r'batch_(\d+)', x).group(1)))

        # Find the highest batch number of encoded files
        encoded_files = [f for f in all_files if
                         f.startswith("works_abstracts_batch_") and f.endswith("_encoded.parquet")]
        if encoded_files:
            highest_encoded_batch = max([int(re.search(r'batch_(\d+)', f).group(1)) for f in encoded_files])
            print(f"Resuming from batch {highest_encoded_batch + 1}")
            start_batch = next((i for i, f in enumerate(batch_files) if
                                int(re.search(r'batch_(\d+)', f).group(1)) > highest_encoded_batch), len(batch_files))
        else:
            print("Starting from the first batch")
            start_batch = 0

        for batch_file in batch_files[start_batch:]:
            input_path = os.path.join(self.output_directory, batch_file)
            output_path = os.path.join(self.output_directory, batch_file.replace(".parquet", "_encoded.parquet"))

            if os.path.exists(output_path):
                print(f"Skipping already processed file: {batch_file}")
                continue

            print(f"Processing file: {batch_file}")

            # Read the parquet file
            df = pd.read_parquet(input_path)

            # Create embeddings
            abstracts = df['abstract_string'].tolist()
            embeddings = []
            for i in tqdm(range(0, len(abstracts), batch_size), desc="Creating embeddings"):
                batch = abstracts[i:i + batch_size]
                batch_embeddings = self.generate_embeddings(model, batch, quantize_embeddings)
                # Convert tensors to numpy arrays
                batch_embeddings = [emb.cpu().numpy() for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)

            # Convert embeddings to a list of numpy arrays
            df['abstract_embedding'] = embeddings

            # Save the updated dataframe
            df.to_parquet(output_path, index=False)
            print(f"Saved processed file: {output_path}")

            # Clear memory
            del df, embeddings
            gc.collect()

        print("All files processed successfully.")

    def collect_combined_data(self, batch_size=1000000, max_works=None):
        client = MongoClient(self.mongo_url)
        db = client[self.mongo_database_name]
        works_collection = db[self.mongo_works_collection_name]

        data = []
        total_processed = 0
        total_with_data = 0
        batch_counter = 0

        for work in tqdm(works_collection.find(), desc="Processing works", unit="work"):
            work_id = work.get('id')
            works_int_id = work.get("works_int_id")
            title = work.get('display_name', '')
            abstract = self.reconstruct_abstract(work.get('abstract_inverted_index', {}))
            has_abstract = bool(abstract)  # New boolean to indicate if abstract exists

            authors = [authorship.get('author', {}).get('display_name', '') for authorship in
                       work.get('authorships', [])]
            authors_string = ' '.join(authors[:20])  # Limit to first 10 authors

            primary_topic = work.get('primary_topic', {})
            if primary_topic:
                field = primary_topic.get('field', {}).get('display_name', '')
                subfield = primary_topic.get('subfield', {}).get('display_name', '')
                topic = primary_topic.get('display_name', '')
            else:
                field = ''
                subfield = ''
                topic = ''

            combined_string = f"{title} {authors_string} {field} {subfield} {topic} | {abstract}".strip()

            if combined_string:
                data.append({
                    'work_id': work_id,
                    "works_int_id": works_int_id,
                    'has_abstract': has_abstract,  # New field
                    'combined_string': combined_string,
                    'field': field,
                    'subfield': subfield,
                    'topic': topic,

                })
                total_with_data += 1

            total_processed += 1

            if len(data) >= batch_size:
                self.save_combined_batch(data, batch_counter)
                data = []
                batch_counter += 1

            if max_works and total_processed >= max_works:
                break

        if data:
            self.save_combined_batch(data, batch_counter)

        client.close()

        print(f"Total works processed: {total_processed}")
        print(f"Works with combined data: {total_with_data}")
    @measure_time
    def save_combined_batch(self, results, batch_counter):
        df = pd.DataFrame(results)
        output_file = os.path.join(self.output_directory, f"works_combined_data_batch_{batch_counter}.parquet")
        df.to_parquet(output_file, index=False)
        print(f"Saved batch {batch_counter} to {output_file}")

    def process_combined_data_with_embeddings(self, batch_size=10_000, quantize_embeddings=False):
        model = SentenceTransformer(self.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        encoded_files = [f for f in os.listdir(self.output_directory) if f.endswith("_combined_encoded.parquet")]
        if encoded_files:
            highest_batch = max([int(re.search(r'batch_(\d+)', f).group(1)) for f in encoded_files])
            start_batch = highest_batch + 1
        else:
            start_batch = 0

        batch_files = sorted([f for f in os.listdir(self.output_directory) if
                              f.startswith("works_combined_data_batch_") and f.endswith(".parquet")],
                             key=lambda x: int(re.search(r'batch_(\d+)', x).group(1)))

        for batch_file in batch_files[start_batch:]:
            input_path = os.path.join(self.output_directory, batch_file)
            output_path = os.path.join(self.output_directory,
                                       batch_file.replace(".parquet", "_combined_encoded.parquet"))

            if os.path.exists(output_path):
                print(f"Skipping already processed file: {batch_file}")
                continue

            print(f"Processing file: {batch_file}")

            df = pd.read_parquet(input_path)
            combined_strings = df['combined_string'].tolist()
            embeddings = []

            for i in tqdm(range(0, len(combined_strings), batch_size), desc="Creating embeddings"):
                batch = combined_strings[i:i + batch_size]
                batch_embeddings = self.generate_embeddings(model, batch, quantize_embeddings)
                batch_embeddings = [emb.cpu().numpy() for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)

            df['combined_embedding'] = embeddings
            df.to_parquet(output_path, index=False)
            print(f"Saved processed file: {output_path}")

            del df, embeddings
            gc.collect()

        print("All combined data files processed successfully.")

    @measure_time
    def generate_embeddings(self, model, abstract_strings, quantize_embeddings=False):
        if quantize_embeddings:
            embeddings = model.encode(abstract_strings, batch_size=64, convert_to_tensor=True, precision="binary",
                                      show_progress_bar=True)
        else:
            embeddings = model.encode(abstract_strings, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
        return embeddings

# Usage
if __name__ == "__main__":
    constructor = CloudDatasetConstruction()
    # constructor.load_and_display_parquet_info()
    # constructor.collect_abstracts()
    # constructor.process_abstracts_with_embeddings()
    constructor.collect_combined_data()
    constructor.process_combined_data_with_embeddings()
    constructor.process_abstracts_with_embeddings()

