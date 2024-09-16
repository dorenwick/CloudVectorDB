import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from span_marker import SpanMarkerModel
import pyarrow as pa
import pyarrow.parquet as pq


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class KeyphraseAcronymExtractor:
    """
    Pls implement all of this.

    """

    def __init__(self, rank, world_size, input_directory, output_directory):
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}'

        self.keyphrase_model = SpanMarkerModel.from_pretrained(
            "tomaarsen/span-marker-bert-base-uncased-keyphrase-inspec")
        self.acronym_model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-base-uncased-acronyms")

        self.keyphrase_model.to(self.device)
        self.acronym_model.to(self.device)

        self.keyphrase_model = DDP(self.keyphrase_model, device_ids=[rank])
        self.acronym_model = DDP(self.acronym_model, device_ids=[rank])

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.progress_file = os.path.join(output_directory, f"extraction_progress_{rank}.json")
        self.input_parquet_file = os.path.join(input_directory, "paragraphs.parquet")
        self.output_parquet_file = os.path.join(output_directory, f"extracted_entities_{rank}.parquet")
        self.processed_paragraphs = self.load_progress()

    def extract_entities(self, text, model):
        entities = model.predict(text)
        return entities

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return 0

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.processed_paragraphs, f)

    def process_paragraphs(self, batch_size=100):
        df = pd.read_parquet(self.input_parquet_file)
        total_paragraphs = len(df)

        # Split the data across GPUs
        chunk_size = total_paragraphs // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank != self.world_size - 1 else total_paragraphs
        df = df.iloc[start_idx:end_idx]

        extracted_data = []

        for i in range(self.processed_paragraphs, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            for _, row in batch.iterrows():
                paragraph_id, paragraph_text, work_id = row['paragraph_id'], row['paragraph_text'], row['work_id']

                if len(paragraph_text) < 10:
                    continue

                keyphrases = self.extract_entities(paragraph_text, self.keyphrase_model)
                acronyms = self.extract_entities(paragraph_text, self.acronym_model)

                for entity in keyphrases:
                    extracted_data.append({
                        'entity_type': 'keyphrase',
                        'entity': entity['span'],
                        'paragraph_id': paragraph_id,
                        'work_id': work_id,
                        'score': entity['score'],
                        'char_start_index': entity['char_start_index'],
                        'char_end_index': entity['char_end_index']
                    })

                for entity in acronyms:
                    extracted_data.append({
                        'entity_type': 'acronym',
                        'entity': entity['span'],
                        'paragraph_id': paragraph_id,
                        'work_id': work_id,
                        'score': entity['score'],
                        'char_start_index': entity['char_start_index'],
                        'char_end_index': entity['char_end_index']
                    })

            self.processed_paragraphs += len(batch)
            self.save_progress()
            print(f"GPU {self.rank}: Processed {self.processed_paragraphs}/{len(df)} paragraphs")

        # Save extracted data to parquet file
        extracted_df = pd.DataFrame(extracted_data)
        extracted_df.to_parquet(self.output_parquet_file, index=False)
        print(f"GPU {self.rank}: Saved extracted entities to {self.output_parquet_file}")

    def run(self):
        self.process_paragraphs()


def run_extractor(rank, world_size, input_directory, output_directory):
    setup(rank, world_size)
    extractor = KeyphraseAcronymExtractor(rank, world_size, input_directory, output_directory)
    extractor.run()
    cleanup()


if __name__ == "__main__":
    world_size = 8  # Number of GPUs
    input_directory = "/path/to/input/directory"  # Replace with your input directory
    output_directory = "/path/to/output/directory"  # Replace with your output directory

    mp.spawn(run_extractor, args=(world_size, input_directory, output_directory), nprocs=world_size, join=True)