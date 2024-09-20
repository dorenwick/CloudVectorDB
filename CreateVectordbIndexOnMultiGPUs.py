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
    GPUs and manages memory limitations.

    import faiss
    import numpy as np

    # Sample data
    embeddings = np.random.random((1000000, 128)).astype('float32')  # 1 million vectors with 128 dimensions

    # List of GPU IDs to use
    gpu_ids = [0, 1]  # Example for using 2 GPUs (GPU 0 and GPU 1)

    # Initialize GPU resources for each GPU
    gpu_resources = []
    for gpu_id in gpu_ids:
        res = faiss.StandardGpuResources()  # Allocate resources for GPU
        gpu_resources.append(res)

    # Create a flat (or any other type) index on CPU
    cpu_index = faiss.IndexFlatL2(128)  # L2 distance, for 128-dimensional vectors

    # Create a sharded GPU index using IndexShards
    sharded_index = faiss.IndexShards(128)  # Shard for dimension size

    # Add each shard for each GPU
    for gpu_id, res in zip(gpu_ids, gpu_resources):
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)  # Move index to GPU
        sharded_index.add_shard(gpu_index)

    # Add embeddings (this will distribute across all shards/GPU)
    sharded_index.add(embeddings)

    # Optionally, train the index (if required by the index type, such as IVF)
    #



    """

    def __init__(self, num_gpus=4, batch_size=1_000_000, embedding_dim=384):
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.gpu_resources = self.initialize_gpu_resources()

    def initialize_gpu_resources(self):
        gpu_resources = []
        for i in range(self.num_gpus):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1 * 1024 * 1024 * 1024)  # 1 GB temporary memory
            gpu_resources.append(res)
        return gpu_resources

    def calculate_index_parameters(self, collection_size):
        if collection_size < 1_000_000:
            nlist = int(4 * math.sqrt(collection_size))
            return f"IVF{nlist},Flat", nlist, None
        elif 1_000_000 <= collection_size < 10_000_000:
            return "IVF65536,HNSW32,Flat", 65536, None
        elif 10_000_000 <= collection_size < 100_000_000:
            return "IVF262144,HNSW32,Flat", 262144, None
        else:  # 100M or more
            return "IVF1048576,HNSW32,PQ16", 1048576, None

    def create_gpu_index(self, d, index_type, nlist, m=None):
        if "PQ" in index_type:
            # For PQ-based index
            coarse_quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, m, 8)
        else:
            # For Flat index
            index = faiss.index_factory(d, index_type)

        # Create a GpuIndexIVFFlat instance for multiple GPUs
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # Shard the index across GPUs
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(
            self.gpu_resources,
            index,
            co,
            gpus=list(range(self.num_gpus))
        )
        return gpu_index

    def train_index(self, index, embeddings):
        index.train(embeddings)

    def add_to_index(self, index, embeddings):
        index.add(embeddings)

    def process_batch(self, batch_data):
        int_ids = [item['work_int_id'] for item in batch_data]
        item_ids = [item['work_id'] for item in batch_data]
        embeddings = np.array([item['embedding'] for item in batch_data])
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

        index_type, nlist, m = self.calculate_index_parameters(total_vectors)
        gpu_index = self.create_gpu_index(self.embedding_dim, index_type, nlist, m)

        all_int_ids = []
        all_item_ids = []

        for i in range(0, len(all_data), self.batch_size):
            batch = all_data[i:i+self.batch_size]
            int_ids, item_ids, embeddings = self.process_batch(batch)

            if i == 0:
                print("Training index...")
                self.train_index(gpu_index, embeddings)

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