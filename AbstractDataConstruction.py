
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from typing import Dict
from typing import Tuple

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

class AbstractDataConstruction():
    """
    python AbstractDataConstruction.py
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
                 batch_size: int = 100_000,
                 extract_keywords: bool = True,
                 generate_embeddings: bool = True):  # New parameter

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        self.extract_keywords = extract_keywords
        self.generate_embeddings_bool = generate_embeddings


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


    def update_ngram_counters(self, df: pd.DataFrame):
        full_text = df['title'] + ' ' + df['authors_string'] + ' ' + df['abstract_string']
        short_text = df['title'] + ' ' + df['authors_string']

        with mp.Pool(processes=mp.cpu_count()) as pool:
            full_unigrams, full_bigrams = pool.map(self.process_ngrams, [(text, 1) for text in full_text])
            short_unigrams, short_bigrams = pool.map(self.process_ngrams, [(text, 1) for text in short_text])

        self.full_unigrams.update(sum(full_unigrams, Counter()))
        self.full_bigrams.update(sum(full_bigrams, Counter()))
        self.short_unigrams.update(sum(short_unigrams, Counter()))
        self.short_bigrams.update(sum(short_bigrams, Counter()))


    def save_ngram_data(self):
        ngram_data = [
            (self.full_unigrams, "full_string_unigrams.parquet"),
            (self.full_bigrams, "full_string_bigrams.parquet"),
            (self.short_unigrams, "short_unigrams.parquet"),
            (self.short_bigrams, "short_bigrams.parquet")
        ]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(self.save_counter,
                         [(counter, file_name, self.output_dir) for counter, file_name in ngram_data])

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

    def calculate_non_zero_counts(self, df: pd.DataFrame):
        df['non_zero_count'] = df['field_count'].apply(lambda x: np.count_nonzero(x))
        return df

    def calculate_ctf_idf_score(self, df: pd.DataFrame):
        B = 26  # Number of fields
        df['ctf_idf_score'] = (B / df['non_zero_count']) / np.log1p(df['count'])
        return df

    def process_ngrams(self):
        print("Starting ngram processing...")

        # Step 1: Read all parquet files and update ngram counters
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])
        for file_name in tqdm(input_files, desc="Processing files for ngrams"):
            input_path = os.path.join(self.input_dir, file_name)
            df = pd.read_parquet(input_path)
            self.update_ngram_counters(df)
            print(f"Processed {file_name} for ngrams")

        # Step 2: Save ngram data
        print("Saving ngram data...")
        self.save_ngram_data()

        # Step 3: Post-process ngram data
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