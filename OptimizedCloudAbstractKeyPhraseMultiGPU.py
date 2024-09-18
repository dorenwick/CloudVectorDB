import os
import time
from typing import List, Dict

import numpy as np
import torch
import pandas as pd
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


class OptimizedCloudAbstractKeyPhraseMultiGPU:
    def __init__(self, input_dir: str,
                 output_dir: str,
                 keyphrase_model_path: str,
                 batch_size: int = 100_000,
                 models_per_gpu: int = 4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.models_per_gpu = models_per_gpu

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
        else:
            print("No GPUs detected. Using CPU.")
            self.devices = ["cpu"]

        # Initialize multiple models on each device
        self.keyphrase_models = []
        for device in self.devices:
            for _ in range(self.models_per_gpu):
                model = SpanMarkerModel.from_pretrained(keyphrase_model_path).to(device)
                model.eval()  # Set the model to evaluation mode
                self.keyphrase_models.append(model)

    def extract_entities(self, texts: List[str], model: SpanMarkerModel) -> List[List[Dict]]:
        with torch.no_grad():  # Disable gradient calculation for inference
            return model.predict(texts)

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        def extract_keywords(sub_batch, model_index):
            model = self.keyphrase_models[model_index]
            device = model.device

            with torch.cuda.device(device):
                sub_batch['keywords_title'] = [[] for _ in range(len(sub_batch))]
                sub_batch['keywords_abstract'] = [[] for _ in range(len(sub_batch))]

                non_empty_titles = [title for title in sub_batch['title'] if isinstance(title, str) and title.strip()]
                if non_empty_titles:
                    title_keywords = self.extract_entities(non_empty_titles, model)
                    for title, keywords in zip(non_empty_titles, title_keywords):
                        idx = sub_batch.index[sub_batch['title'] == title].tolist()
                        if idx:
                            sub_batch.at[idx[0], 'keywords_title'] = keywords

                non_empty_abstracts = [abstract for abstract in sub_batch['abstract_string'] if
                                       isinstance(abstract, str) and abstract.strip()]
                if non_empty_abstracts:
                    abstract_keywords = self.extract_entities(non_empty_abstracts, model)
                    for abstract, keywords in zip(non_empty_abstracts, abstract_keywords):
                        idx = sub_batch.index[sub_batch['abstract_string'] == abstract].tolist()
                        if idx:
                            sub_batch.at[idx[0], 'keywords_abstract'] = keywords

            return sub_batch

        # Split the batch across available models
        sub_batches = np.array_split(batch, len(self.keyphrase_models))

        # Use ThreadPoolExecutor to run extraction on multiple models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.keyphrase_models)) as executor:
            futures = [executor.submit(extract_keywords, sub_batch, i) for i, sub_batch in enumerate(sub_batches)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Combine results
        return pd.concat(results, ignore_index=True)

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
        print(entity_df.head(10).to_string())
        print("length dataframe: ", len(entity_df))
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
                    print(f"Skipping {file_name} as it has already been processed.")
                    continue

                df = pd.read_parquet(input_path)
                processed_df = self.process_batch(df)
                self.save_entity_data(processed_df)

                # Print progress information
                print(f"Processed {file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    def run(self):
        self.process_files()
        print("Keyphrase extraction completed successfully.")


if __name__ == "__main__":
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    keyphrase_model_path = "/workspace/models/models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec/snapshots/bfc31646972e22ebf331c2e877c30439f01d35b3"

    processor = OptimizedCloudAbstractKeyPhraseMultiGPU(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path
    )
    processor.run()