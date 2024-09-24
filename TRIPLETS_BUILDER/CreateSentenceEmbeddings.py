import os
import torch
from tqdm import tqdm
import polars as pl
from sentence_transformers import SentenceTransformer
from functools import wraps
import time

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

class CreateSentenceEmbeddings:
    def __init__(self, model_path, works_all_collected_file, embeddings_directory, matryoshka_dim=12):
        self.model_path = model_path
        self.works_all_collected_file = works_all_collected_file
        self.embeddings_directory = embeddings_directory
        self.matryoshka_dim = matryoshka_dim

    @measure_time
    def create_sentence_embeddings(self, works_batch_size=100_000):
        works_file = self.works_all_collected_file
        df = pl.read_parquet(works_file)
        total_works = len(df)
        total_batches = (total_works + works_batch_size - 1) // works_batch_size

        # Load the Matryoshka model with truncated dimensions
        model = self.load_matryoshka_model(self.model_path, self.matryoshka_dim)

        # Check for multiple GPUs and wrap in DataParallel if necessary
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

        model.to('cuda')

        for batch_num in range(total_batches):
            torch.cuda.empty_cache()
            start_idx = batch_num * works_batch_size
            end_idx = min((batch_num + 1) * works_batch_size, total_works)
            batch_works = df.slice(start_idx, end_idx - start_idx)

            sentences = []
            work_ids = []
            work_int_ids = []

            for work in batch_works.iter_rows(named=True):
                sentence = self.create_sentence_work(work)
                sentences.append(sentence)
                work_ids.append(work['work_id'])
                work_int_ids.append(work['work_int_id'])

            with torch.no_grad():
                # Process sentences in batches of 64 for encoding
                embeddings = []
                for i in tqdm(range(0, len(sentences), 64), desc=f"Encoding batch {batch_num + 1}/{total_batches}"):
                    batch = sentences[i:i + 64]

                    # Use model.encode to get embeddings
                    if isinstance(model, torch.nn.DataParallel):
                        batch_embeddings = model.module.encode(batch, convert_to_tensor=True,
                                                               device='cuda').cpu().numpy()
                    else:
                        batch_embeddings = model.encode(batch, convert_to_tensor=True, device='cuda').cpu().numpy()

                    embeddings.extend(batch_embeddings)

            # Create a DataFrame to store the results
            batch_data = pl.DataFrame({
                'work_id': work_ids,
                'work_int_id': work_int_ids,
                'work_sentence': sentences,
                'work_embedding': embeddings
            })

            # Save the batch to a parquet file
            file_name = f'work_embeddings_batch_{batch_num}.parquet'
            file_path = os.path.join(self.embeddings_directory, file_name)
            batch_data.write_parquet(file_path)

            print(f"Processed batch {batch_num + 1}/{total_batches}, saved to {file_path}")

        print(f"Sentence embeddings created and saved in {self.embeddings_directory}")
        print(f"Total works processed: {total_works}")

    def load_matryoshka_model(self, model_path, matryoshka_dim):
        model = SentenceTransformer(model_path, truncate_dim=matryoshka_dim)
        return model

    def create_sentence_work(self, work_info):
        display_name = work_info.get('title_string', '')
        authors_string = work_info.get('authors_string', '')
        field = work_info.get('field_string', '')
        subfield = work_info.get('subfield_string', '')

        query_string = f"{display_name} {authors_string} {field} {subfield}"
        return query_string


    @measure_time
    def sort_files_numerically(self):
        files = os.listdir(self.embeddings_directory)
        parquet_files = [f for f in files if f.endswith('.parquet') and '_embeddings' in f]
        unique_files = list(set(parquet_files))  # Remove duplicates
        sorted_files = sorted(unique_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return sorted_files



embeddings_creator = CreateSentenceEmbeddings(
    model_path='path/to/model',
    works_all_collected_file='path/to/works_all_collected.parquet',
    embeddings_directory='path/to/embeddings/directory',
    matryoshka_dim=12
)

embeddings_creator.create_sentence_embeddings(works_batch_size=100_000)