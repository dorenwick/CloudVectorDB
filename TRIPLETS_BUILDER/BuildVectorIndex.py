import os
import math
import numpy as np
import faiss
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm
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

class BuildVectorIndex:
    def __init__(self, embeddings_directory, vectordb_directory, num_gpus):
        self.embeddings_directory = embeddings_directory
        self.vectordb_directory = vectordb_directory
        self.num_gpus = num_gpus
        self.gpu_resources = self.initialize_gpu_resources()

    @measure_time
    def sort_files_numerically(self):
        files = os.listdir(self.embeddings_directory)
        parquet_files = [f for f in files if f.endswith('.parquet') and '_embeddings' in f]
        unique_files = list(set(parquet_files))  # Remove duplicates
        sorted_files = sorted(unique_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return sorted_files

    @measure_time
    def build_vector_index(self, N=20_000_000, use_gpu=True):
        sorted_files = self.sort_files_numerically()

        all_data = []
        total_records = 0

        for file in tqdm(sorted_files, desc="Loading data"):
            file_path = os.path.join(self.embeddings_directory, file)
            print(f"Processing file: {file_path}")
            print(f"Current records: {len(all_data)}")
            table = pq.read_table(file_path)
            data = table.to_pandas()
            all_data.extend(data.to_dict('records'))
            total_records += len(data)
            if total_records >= N:
                break

        print(f"Total number of records loaded: {len(all_data)}")

        work_int_ids = [item['work_int_id'] for item in all_data]
        work_ids = [item['work_id'] for item in all_data]
        embeddings = np.array([item['work_embedding'] for item in all_data])

        print(f"Shape of embeddings: {embeddings.shape}")

        d = embeddings.shape[1]
        n = embeddings.shape[0]
        hnsw_m = 32

        index_type, nlist = self.calculate_index_parameters(n)

        print("index_type, nlist, hnsw_m", index_type, nlist, hnsw_m)

        if use_gpu:
            index = self.train_index_gpu(embeddings, work_int_ids, d, index_type, nlist, hnsw_m)
        else:
            index = self.train_index_cpu(embeddings, work_int_ids, d, index_type, nlist, hnsw_m)

        nlist_num = int(math.sqrt(nlist))
        nprobe_count = min(128, nlist_num)
        print("nprobe_count ", nprobe_count)

        index.nprobe = nprobe_count

        index_path = os.path.join(self.vectordb_directory, "works_index.bin")
        faiss.write_index(index, index_path)

        mapping_df = pl.DataFrame({
            'works_int_id': work_int_ids,
            'work_id': work_ids,
        })

        mapping_path = os.path.join(self.vectordb_directory, "works_id_mapping.parquet")
        mapping_df.write_parquet(mapping_path)

        print(f"FAISS index created and saved to {index_path}")
        print(f"ID mapping saved to {mapping_path}")

        return index_path, mapping_path

    def initialize_gpu_resources(self):
        gpu_resources = []
        for i in range(self.num_gpus):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1 * 1024 * 1024 * 1024)  # 1 GB temporary memory
            gpu_resources.append(res)
        return gpu_resources

    def calculate_index_parameters(self, collection_size):
        if collection_size < 1_000_000:
            nlist = 8 * int(4 * math.sqrt(collection_size))
            return f"IVF{nlist},Flat", nlist
        elif 1_000_000 <= collection_size < 10_000_000:
            return "IVF65536,Flat", 65536
        elif 10_000_000 <= collection_size < 25_000_000:
            return "IVF262144,Flat", 262144
        else:  # 25M or more
            return "IVF1048576,Flat", 1048576

    @measure_time
    def train_index_gpu(self, embeddings, work_int_ids, d, index_type, nlist, hnsw_m):
        print(f"Training GPU index with {index_type}")

        # Create the index
        index = faiss.index_factory(d, index_type)

        # Convert to GPU index
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(self.gpu_resources, index, co)

        # Train the index
        gpu_index.train(embeddings)

        # Create an IndexIDMap to store custom IDs
        cpu_index = faiss.IndexIDMap(faiss.index_gpu_to_cpu(gpu_index))

        # Add vectors to the index with custom IDs
        cpu_index.add_with_ids(embeddings, np.array(work_int_ids))

        return cpu_index

    @measure_time
    def train_index_cpu(self, embeddings, work_int_ids, d, index_type, nlist, hnsw_m):
        print(f"Training CPU index with {index_type}")
        index = faiss.index_factory(d, index_type)
        index.train(embeddings)

        # Create an IndexIDMap to store custom IDs
        id_map_index = faiss.IndexIDMap(index)

        # Add vectors to the index with custom IDs
        id_map_index.add_with_ids(embeddings, np.array(work_int_ids))

        return id_map_index

    @measure_time
    def load_index_and_mapping(self):
        index_path = os.path.join(self.vectordb_directory, "works_index.bin")
        self.vector_index = faiss.read_index(index_path)
        mapping_path = os.path.join(self.vectordb_directory, "works_id_mapping.parquet")
        self.faiss_to_work_id_mapping = pl.read_parquet(mapping_path)



index_builder = BuildVectorIndex(
    embeddings_directory='path/to/embeddings',
    vectordb_directory='path/to/vectordb',
    num_gpus=2
)

index_path, mapping_path = index_builder.build_vector_index(N=20_000_000, use_gpu=True)