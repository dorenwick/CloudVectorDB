import os
import math
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
import torch

class CreateVectordbIndexOnMultiGPUs:
    """
    This class is used to create a vectordb index by loading the faiss embeddings
    onto multiple A100 GPUs (4 or 8). It handles the distribution of work across
    GPUs and manages memory limitations. It uses Hierarchical Navigable Small Worlds (HNSW) for indexing.
    """

    def __init__(self, num_gpus=4, batch_size=1_000_000, embedding_dim=384, ef_search=200, ef_construction=100):
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.gpu_resources = self.initialize_gpu_resources()

    def initialize_gpu_resources(self):
        gpu_resources = []
        for i in range(self.num_gpus):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1 * 1024 * 1024 * 1024)  # 1 GB temporary memory
            gpu_resources.append(res)
        return gpu_resources

    def create_gpu_index(self):
        # HNSW index: IndexHNSWFlat does not need training
        cpu_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 neighbors per layer in HNSW

        # Set HNSW parameters for the index
        cpu_index.hnsw.efConstruction = self.ef_construction  # Control memory/speed tradeoff during construction
        cpu_index.hnsw.efSearch = self.ef_search  # Control accuracy of the search

        # Convert the CPU index to a multi-GPU index
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(
            self.gpu_resources,
            cpu_index,
            gpus=list(range(self.num_gpus))
        )

        return gpu_index

    def add_to_index(self, index, embeddings):
        index.add(embeddings)

    def process_batch(self, batch_data):
        int_ids = [item['work_int_id'] for item in batch_data]
        item_ids = [item['work_id'] for item in batch_data]
        embeddings = np.array([item['embedding'] for item in batch_data], dtype=np.float32)
        return int_ids, item_ids, embeddings

    def create_index(self, embedding_csv_directory, output_directory, collection_name, max_vectors=None):
        sorted_files = sorted(
            [f for f in os.listdir(embedding_csv_directory) if f.endswith('.parquet')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        all_data = []
        total_vectors = 0

        for file in sorted_files:
            file_path = os.path.join(embedding_csv_directory, file)
            table = pq.read_table(file_path)
            data = table.to_pandas()
            all_data.extend(data.to_dict('records'))
            total_vectors += len(data)

            if max_vectors and total_vectors >= max_vectors:
                all_data = all_data[:max_vectors]
                total_vectors = max_vectors
                break

        print(f"Total number of vectors to process: {total_vectors}")

        # Create HNSW index on GPUs
        gpu_index = self.create_gpu_index()

        all_int_ids = []
        all_item_ids = []

        for i in range(0, len(all_data), self.batch_size):
            batch = all_data[i:i+self.batch_size]
            int_ids, item_ids, embeddings = self.process_batch(batch)

            print(f"Adding batch {i//self.batch_size + 1} to index...")
            self.add_to_index(gpu_index, embeddings)

            all_int_ids.extend(int_ids)
            all_item_ids.extend(item_ids)

            del embeddings
            gc.collect()
            torch.cuda.empty_cache()

        # Convert GPU index back to CPU for saving
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)

        index_path = os.path.join(output_directory, f"index_{collection_name.lower()}.bin")
        faiss.write_index(cpu_index, index_path)

        mapping_df = pd.DataFrame({
            'int_ids': all_int_ids,
            f'{collection_name.lower()}_ids': all_item_ids,
        })
        mapping_path = os.path.join(output_directory, f"id_mapping_{collection_name.lower()}.csv")
        mapping_df.to_csv(mapping_path, index=False)

        print(f"Index created and saved for {collection_name}.")
        return cpu_index, mapping_df

    def add_vectors_to_index(self, index_path, mapping_path, new_embedding_files, collection_name):
        cpu_index = faiss.read_index(index_path)
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(
            self.gpu_resources,
            cpu_index,
            gpus=list(range(self.num_gpus))
        )
        mapping_df = pd.read_csv(mapping_path)

        for file in new_embedding_files:
            table = pq.read_table(file)
            data = table.to_pandas()

            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size].to_dict('records')
                int_ids, item_ids, embeddings = self.process_batch(batch)

                self.add_to_index(gpu_index, embeddings)

                new_mapping = pd.DataFrame({
                    'int_ids': int_ids,
                    f'{collection_name.lower()}_ids': item_ids,
                })
                mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)

                del embeddings
                gc.collect()
                torch.cuda.empty_cache()

        # Convert back to CPU index for saving
        updated_cpu_index = faiss.index_gpu_to_cpu(gpu_index)

        faiss.write_index(updated_cpu_index, index_path)
        mapping_df.to_csv(mapping_path, index=False)

        print(f"New vectors added to the index for {collection_name}.")
        return updated_cpu_index, mapping_df
