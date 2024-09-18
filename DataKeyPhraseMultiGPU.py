import os
import json
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from span_marker import SpanMarkerModel
from tqdm import tqdm


class CloudKeyphraseExtractor:
    def __init__(self, keyphrase_model_name="tomaarsen/span-marker-bert-base-uncased-keyphrase-inspec",
                 progress_file="/workspace/extraction_progress.json",
                 input_parquet_file="/workspace/paragraphs/paragraphs.parquet",
                 output_parquet_file="/workspace/results/keyphrases.parquet",
                 batch_size=100,
                 start_index=1_000_000):
        self.keyphrase_model_name = keyphrase_model_name
        self.progress_file = progress_file
        self.input_parquet_file = input_parquet_file
        self.output_parquet_file = output_parquet_file
        self.batch_size = batch_size
        self.start_index = start_index
        self.processed_paragraphs = self.load_progress()
        self.setup_gpus()

    def setup_gpus(self):
        self.devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        if not self.devices:
            self.devices = ['cpu']
        print(f"Using devices: {self.devices}")

        self.models = [SpanMarkerModel.from_pretrained(self.keyphrase_model_name).to(device)
                       for device in self.devices]

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return self.start_index

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.processed_paragraphs, f)

    def extract_entities(self, texts):
        results = []
        for i, text in enumerate(texts):
            model = self.models[i % len(self.models)]
            device = self.devices[i % len(self.devices)]
            with torch.cuda.device(device):
                entities = model.predict(text)
                results.extend(entities)
        return results

    def process_paragraphs(self):
        df = pd.read_parquet(self.input_parquet_file)
        total_paragraphs = len(df)
        results = []

        for i in tqdm(range(self.processed_paragraphs, total_paragraphs, self.batch_size)):
            batch = df.iloc[i:i + self.batch_size]

            keyphrases = self.extract_entities(batch['paragraph_text'].tolist())

            for (_, row), keyphrase_list in zip(batch.iterrows(), keyphrases):
                for keyphrase in keyphrase_list:
                    results.append({
                        'keyphrase': keyphrase['span'],
                        'paragraph_id': row['paragraph_id'],
                        'work_id': row['work_id'],
                        'score': keyphrase['score'],
                        'char_start_index': keyphrase['char_start_index'],
                        'char_end_index': keyphrase['char_end_index']
                    })

            self.processed_paragraphs += len(batch)
            self.save_progress()

            if len(results) >= 10000:
                self.save_results(results)
                results = []

        if results:
            self.save_results(results)

    def save_results(self, results):
        df = pd.DataFrame(results)
        if os.path.exists(self.output_parquet_file):
            existing_df = pd.read_parquet(self.output_parquet_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_parquet(self.output_parquet_file, index=False)
        print(f"Saved results to {self.output_parquet_file}")

    def run(self):
        self.process_paragraphs()
        print("Keyphrase extraction completed successfully.")


if __name__ == "__main__":
    extractor = CloudKeyphraseExtractor()
    extractor.run()