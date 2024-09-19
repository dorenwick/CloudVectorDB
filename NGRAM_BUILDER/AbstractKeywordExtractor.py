import os
import time
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
from span_marker import SpanMarkerModel
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

class AbstractKeywordExtractor:
    def __init__(self, input_dir: str, output_dir: str, keyphrase_model_path: str, batch_size: int = 100_000):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size

        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("CUDA device name:", torch.cuda.get_device_name(0))

        # Set up GPU or fall back to CPU
        self.device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the keyphrase model
        self.keyphrase_model = SpanMarkerModel.from_pretrained(keyphrase_model_path).to(self.device)

    def extract_entities(self, texts: List[str]) -> List[List[Dict]]:
        with torch.cuda.device(self.device):
            return self.keyphrase_model.predict(texts)

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        try:
            # Process non-empty abstracts
            non_empty_abstracts = [abstract for abstract in batch['abstract_string'] if
                                   isinstance(abstract, str) and abstract.strip()]
            if non_empty_abstracts:
                abstract_keywords = self.extract_entities(non_empty_abstracts)
                keyword_data = []
                for abstract, keywords in zip(non_empty_abstracts, abstract_keywords):
                    work_id = batch.loc[batch['abstract_string'] == abstract, 'work_id'].iloc[0]
                    keyword_data.append({
                        'work_id': work_id,
                        'abstract_string': abstract,
                        'keywords_abstract': keywords
                    })
                return pd.DataFrame(keyword_data)
            else:
                return pd.DataFrame(columns=['work_id', 'abstract_string', 'keywords_abstract'])
        except Exception as e:
            print(f"Error in process_batch: {str(e)}")
            return pd.DataFrame(columns=['work_id', 'abstract_string', 'keywords_abstract'])

    @measure_time
    def save_processed_batch(self, df: pd.DataFrame, output_path: str):
        df.to_parquet(output_path, index=False)

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.startswith('works_combined_data_batch_') and f.endswith('.parquet')])

        for file_name in tqdm(input_files, desc="Processing files for keyword extraction"):
            try:
                input_path = os.path.join(self.input_dir, file_name)
                output_path = os.path.join(self.output_dir, f"keyphrase_abstract_batch_{file_name}")

                if os.path.exists(output_path):
                    print(f"Skipping {file_name} as it has already been processed.")
                    continue

                df = pd.read_parquet(input_path)
                processed_df = self.process_batch(df)
                self.save_processed_batch(processed_df, output_path)

                print(f"Processed keywords for {file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    def run(self):
        print("Starting keyword extraction...")
        self.process_files()
        print("Keyword extraction completed successfully.")

if __name__ == "__main__":
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    keyphrase_model_path = "/workspace/models/models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec/snapshots/bfc31646972e22ebf331c2e877c30439f01d35b3"

    extractor = AbstractKeywordExtractor(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path
    )
    extractor.run()