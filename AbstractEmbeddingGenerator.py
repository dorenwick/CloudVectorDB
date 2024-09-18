
import os
import time
from typing import List
import numpy as np
import pandas as pd
import torch
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

class AbstractEmbeddingGenerator:
    def __init__(self, input_dir: str, output_dir: str, embedding_model_path: str, batch_size: int = 100_000):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("CUDA device name:", torch.cuda.get_device_name(0))

        # Set up GPU or fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_path).to(self.device)

    @measure_time
    def generate_embeddings(self, texts: List[str], quantize_embeddings: bool = False) -> np.ndarray:
        with torch.cuda.device(self.device):
            if quantize_embeddings:
                embeddings = self.embedding_model.encode(texts, batch_size=512, convert_to_tensor=True,
                                                         precision="binary", show_progress_bar=True)
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
        abstract_string_embeddings = self.generate_embeddings(batch['abstract_string'].tolist())
        batch['abstract_embedding'] = list(abstract_string_embeddings)
        return batch[['work_id', 'abstract_string', 'abstract_embedding']]

    def save_processed_batch(self, df: pd.DataFrame, output_path: str):
        df.to_parquet(output_path, index=False)

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.startswith('works_combined_data_batch_') and f.endswith('.parquet')])

        for file_name in tqdm(input_files, desc="Processing files"):
            try:
                input_path = os.path.join(self.input_dir, file_name)
                output_path = os.path.join(self.output_dir, f"embeddings_{file_name}")

                if os.path.exists(output_path):
                    print(f"Skipping {file_name} as it has already been processed.")
                    continue

                df = pd.read_parquet(input_path)
                processed_df = self.process_batch(df)
                self.save_processed_batch(processed_df, output_path)

                print(f"Processed {file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    def run(self):
        print("Starting embedding generation...")
        self.process_files()
        print("Embedding generation completed successfully.")

if __name__ == "__main__":
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    embedding_model_path = "/workspace/models/models--Snowflake--snowflake-arctic-embed-xs/snapshots/86a07656cc240af5c7fd07bac2f05baaafd60401"

    generator = AbstractEmbeddingGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        embedding_model_path=embedding_model_path
    )
    generator.run()