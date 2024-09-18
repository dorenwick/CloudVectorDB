import time

import os
import logging
from typing import List, Dict

import torch
import pandas as pd
from span_marker import SpanMarkerModel
from tqdm import tqdm
from accelerate import PartialState
from accelerate.inference import prepare_pippy


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper




class RefactoredOptimizedCloudAbstractKeyPhraseMultiGPU:
    def __init__(self, input_dir: str, output_dir: str, keyphrase_model_path: str, batch_size: int = 32):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.distributed_state = PartialState()

        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")

        # Load model
        self.model = SpanMarkerModel.from_pretrained(keyphrase_model_path)
        self.model.eval()

        # Prepare model for pipeline parallelism
        example_input = torch.randint(0, 1000, (1, 128), dtype=torch.long)
        self.model = prepare_pippy(self.model, example_args=(example_input,))

    def extract_entities(self, texts: List[str]) -> List[List[Dict]]:
        with torch.no_grad():
            return self.model.predict(texts)

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Convert DataFrame to a dictionary of lists
        batch_dict = {col: batch[col].tolist() for col in batch.columns}

        with self.distributed_state.split_between_processes(batch_dict) as distributed_batch_dict:
            # Convert the distributed dictionary back to a DataFrame
            distributed_batch = pd.DataFrame(distributed_batch_dict)

            distributed_batch['keywords_title'] = [[] for _ in range(len(distributed_batch))]
            distributed_batch['keywords_abstract'] = [[] for _ in range(len(distributed_batch))]

            for column in ['title', 'abstract_string']:
                non_empty_texts = [text for text in distributed_batch[column] if isinstance(text, str) and text.strip()]
                if non_empty_texts:
                    keywords = self.extract_entities(non_empty_texts)
                    for text, kw in zip(non_empty_texts, keywords):
                        idx = distributed_batch.index[distributed_batch[column] == text].tolist()
                        if idx:
                            distributed_batch.at[idx[0], f'keywords_{column.split("_")[0]}'] = kw

        # Gather results from all processes
        all_results = self.distributed_state.gather(distributed_batch)

        # Combine results if on the main process
        if self.distributed_state.is_main_process:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame for non-main processes

    def save_entity_data(self, df: pd.DataFrame):
        entity_data = []
        for _, row in df.iterrows():
            work_id = row['work_id']
            for location in ['title', 'abstract']:
                entities = row[f'keywords_{location}']
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
        output_path = os.path.join(self.output_dir, "keywords_data.parquet")
        if os.path.exists(output_path):
            existing_df = pd.read_parquet(output_path)
            entity_df = pd.concat([existing_df, entity_df], ignore_index=True)
        entity_df.to_parquet(output_path, index=False)

    def process_files(self):
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.parquet')])

        for file_name in tqdm(input_files, desc="Processing files"):
            try:
                input_path = os.path.join(self.input_dir, file_name)
                output_path = os.path.join(self.output_dir, f"keywords_{file_name}")

                if os.path.exists(output_path):
                    logging.info(f"Skipping {file_name} as it has already been processed.")
                    continue

                df = pd.read_parquet(input_path)
                for i in range(0, len(df), self.batch_size):
                    batch = df.iloc[i:i + self.batch_size]
                    processed_batch = self.process_batch(batch)
                    self.save_entity_data(processed_batch)

                logging.info(f"Processed {file_name}")

            except Exception as e:
                logging.error(f"Error processing file {file_name}: {e}")

    def run(self):
        self.process_files()
        logging.info("Keyphrase extraction completed successfully.")


if __name__ == "__main__":
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    keyphrase_model_path = "/workspace/models/models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec/snapshots/bfc31646972e22ebf331c2e877c30439f01d35b3"

    processor = RefactoredOptimizedCloudAbstractKeyPhraseMultiGPU(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path
    )
    processor.run()
