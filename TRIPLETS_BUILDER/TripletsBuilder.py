import concurrent.futures
import gc
import json
import math
import os
import platform
import random
import time
from collections import Counter
from itertools import combinations

import faiss
import numpy as np
import polars as pl
import psutil
import pyarrow.parquet as pq
import torch
from pylatexenc.latex2text import LatexNodes2Text
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from TRIPLETS_BUILDER.AugmentData import AugmentData

latex = "Your LaTeX code here"
text = LatexNodes2Text().latex_to_text(latex)

from BuildVectorIndex import BuildVectorIndex
from CollectAllWorksMetaData import CollectAllWorksMetadata
from CreateSentenceEmbeddings import CreateSentenceEmbeddings


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper



class CloudDatasetConstructionSentenceEncoderT1:
    """



    TODO: rewrite this all so that we instead switch to a system where we create hard negative triplets instead of doing all
        of the stuff with creating pairs

            What we want to do here is this:

            'work_id_one': work1_id,
            'full_string_one': f"{work1.get('title_string', '')} {work1.get('authors_string', '')} {work1.get('field_string', '')} {work1.get('subfield_string', '')}",
            'work_id_two': work2_id,
            'full_string_two': f"{work2.get('title_string', '')} {work2.get('authors_string', '')} {work2.get('field_string', '')} {work2.get('subfield_string', '')}",
            'common_unigrams': vectorized_unigrams[i],
            'common_bigrams': vectorized_bigrams[i],
            'common_field': bool(vectorized_fields[i]),
            'common_subfield': bool(vectorized_subfields[i]),
            'total_score': 0.0,
            'label': '',
            'label_int': 0,
            'p_value': 0.0

            common_authors_file_filtered = os.path.join(self.datasets_directory, "works_common_authors_filtered.parquet")

            those are the contents of the author pairs. What we will do is, for each work_id_one, we will
            go to the mapping_df that contains work_int_id and work_id, and index the work_id, then retreive
            the work_int_id of that work_id from doing that. Then, we will
            encode the vector of full_string_one, and do a k=128 nearest neighbour search over our vector database
            and index, and we will retrieve the results.

            Out of these results, we will review each of the top 128 results as potential triplets.
            we do this by comparing each result with the encoding of full_string_one, and full_string_two,
            (so get distance scores of them), and then we also calculate the common field, common_subfield
            So, we essentially are making 128*2 pairs, and finding the best candidate for each work.
            It might pay to look at how this is done with the generate_training_pairs method we currently have.
            That method shows how to batch process this.

            So, here are some guidelines:
            of the 128 results, we compute the embedding similarity of all the 128*2 pairs, and take the common_unigrams,
            common_bigrams, common_field, common_subfield as well.
            We filter out any pairs that have distance scores lower than 0.15 and any pairs that have distance scores higher than 0.3
            The criterion for triplet selection is this:
            find the triplet where each pair has distance scores between 0.15 and 0.3, and the positive and negative have the highest total_score
            that is still lower than the total_score between anchor and negative,
            where work_id_one is the anchor, the positive is work_id_two, and the negative is the one with distance scores between 0.15 and 0.3
            to both the anchor and positive, and the highest total score to anchor that is still lower than the total score to positive.

            Then, we will want to run this with batch processing techniques.

            Remember that we are running this all via the

            We filter out


        if self.run_params.get('collect_all_works_metadata', False):
            self.collect_all_works_metadata(abstract_include=False)

        if self.run_params.get('create_sentence_embeddings', False):
            self.create_sentence_embeddings(works_batch_size=100_000)

        if self.run_params.get('build_vector_index', False):
            self.build_vector_index(N=20_000_000, use_gpu=True)

        if self.run_params.get('restructure_common_authors', False):
            self.restructure_common_authors()

        if self.run_params.get('restructure_augmented_data', False):
            self.restructure_augmented_data(generate_all_augmentations=False)

        if self.run_params.get('generate_training_pairs', False):
            self.generate_training_pairs(batch_size=4096, knn=128, distance_threshold=0.1, min_count=3, max_appearances=8)




    """

    def __init__(self,
                 model_path,
                 output_directory,
                 datasets_directory,
                 embeddings_directory,
                 ngrams_directory,
                 vectordb_directory,
                 run_params,
                 num_knn_pairs=500_000_000,
                 num_works_collected=500_000_000,
                 matryoshka_dim=12,
                 mongo_url="mongodb://localhost:27017/",
                 mongo_database_name="OpenAlex",
                 mongo_works_collection_name="Works"):

        self.model_path = model_path
        self.num_knn_pairs = num_knn_pairs
        self.num_works_collected = num_works_collected

        self.datasets_directory = datasets_directory
        self.output_directory = output_directory
        self.embeddings_directory = embeddings_directory
        self.ngrams_directory = ngrams_directory
        self.vectordb_directory = vectordb_directory

        self.run_params = run_params

        # TODO: what is difference between self.works_knn_search_file and self.works_all_collected_file ??

        # File paths
        self.works_all_collected_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        self.works_common_authors_file = os.path.join(self.datasets_directory, "works_common_authors.parquet")
        self.works_common_authors_filtered_file = os.path.join(self.datasets_directory, "works_common_authors_filtered.parquet")
        self.works_common_titles_file = os.path.join(self.datasets_directory, "common_title_works.parquet")
        self.works_knn_search_file = os.path.join(self.datasets_directory, "works_knn_search.parquet")
        self.works_augmented_data_file = os.path.join(self.datasets_directory, "works_augmented_data.parquet")
        self.triplet_work_ids_only_file = os.path.join(self.datasets_directory, "triplet_work_ids_only.parquet")
        self.id_mapping_works_file = os.path.join(self.vectordb_directory, "works_id_mapping.parquet")
        self.index_works_file = os.path.join(self.vectordb_directory, "works_index.bin")
        self.triplets_file = os.path.join(self.datasets_directory, "triplets.parquet")
        self.unigram_data_file = os.path.join(self.ngrams_directory, "unigram_data.parquet")
        self.bigram_data_file = os.path.join(self.ngrams_directory, "bigram_data.parquet")

        # MongoDB connection
        self.mongo_url = mongo_url
        self.mongo_database_name = mongo_database_name
        self.mongo_works_collection_name = mongo_works_collection_name
        self.mongo_client = None
        self.mongo_db = None
        self.mongodb_works_collection = None


        self.model = self.load_matryoshka_model(self.model_path, matryoshka_dim)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Other initializations
        self.work_id_search_count = {}
        self.works_knn_search = []
        self.vector_index = None
        self.faiss_to_work_id_mapping = None
        self.works_df = None
        self.work_details = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        self.gpu_resources = self.initialize_gpu_resources()

        # Initialize the new classes
        self.collect_metadata = CollectAllWorksMetadata(
            datasets_directory=datasets_directory,
            ngrams_directory=ngrams_directory,
            mongo_url=mongo_url,
            mongo_database_name=mongo_database_name,
            mongo_works_collection_name=mongo_works_collection_name,
            num_works_collected=num_works_collected
        )

        self.create_embeddings = CreateSentenceEmbeddings(
            model_path=model_path,
            works_all_collected_file=self.works_all_collected_file,
            embeddings_directory=embeddings_directory,
            matryoshka_dim=matryoshka_dim
        )

        self.build_index = BuildVectorIndex(
            embeddings_directory=embeddings_directory,
            vectordb_directory=vectordb_directory,
            num_gpus=self.num_gpus
        )

        self.augment_data = AugmentData(self.datasets_directory, self.ngrams_directory)

    def establish_mongodb_connection(self):
        self.mongo_client = MongoClient(self.mongo_url)
        self.mongo_db = self.mongo_client[self.mongo_database_name]
        self.mongodb_works_collection = self.mongo_db[self.mongo_works_collection_name]

    def close_mongodb_connection(self):
        if self.mongo_client:
            self.mongo_client.close()
            self.mongo_client = None
            self.mongo_db = None
            self.mongodb_works_collection = None
    @measure_time
    def run(self):
        if self.run_params.get('collect_all_works_metadata', False):
            self.collect_metadata.collect_all_works_metadata(abstract_include=False)

        if self.run_params.get('create_sentence_embeddings', False):
            self.create_embeddings.create_sentence_embeddings(works_batch_size=100_000)

        if self.run_params.get('build_vector_index', False):
            self.build_index.build_vector_index(N=20_000_000, use_gpu=True)

        if self.run_params.get('restructure_common_authors', False):
            self.restructure_common_authors(max_author_count=-1)

        if self.run_params.get('generate_training_pairs', False):
            self.generate_training_pairs(batch_size=4096, knn=128, distance_threshold=0.1, min_count=3,
                                         max_appearances=8)

        if self.run_params.get('restructure_augmented_data', False):
            self.augment_data.restructure_augmented_data(generate_all_augmentations=False)

        if self.run_params.get('generate_all_work_id_pairs_dataset', False):
            self.generate_all_work_id_pairs_dataset(sort_by_distance=True)


    def print_memory_usage(self, location):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage at {location}: {memory_info.rss / 1024 / 1024:.2f} MB")


    @measure_time
    def restructure_common_authors(self, max_author_count=-1):
        """
        TODO: hard-coding is bad. eventually we want to deal with this.


        :param max_author_count:
        :return:
        """


        print("Restructuring common authors file...")

        # Load domain to field mapping
        with open('C:\\Users\\doren\\PycharmProjects\\CloudVectorDB\\TRIPLETS_BUILDER\\domain_to_field.json', 'r') as f:
            domain_to_field = json.load(f)
        field_to_domain = {field: domain for domain, fields in domain_to_field.items() for field in fields}

        common_authors_file = os.path.join(self.datasets_directory, "works_common_authors.parquet")
        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")

        print("Reading common authors file...")
        df = pl.read_parquet(common_authors_file)

        print("Reading works file...")
        works_df = pl.read_parquet(works_file)

        initial_rows = df.shape[0]
        print(f"Initial number of rows: {initial_rows}")

        print("Removing duplicate work_id pairs...")
        df_filtered = df.filter(pl.col("work_id_one") != pl.col("work_id_two"))

        # Create a set to keep track of encountered work_ids
        encountered_work_ids = set()

        # Function to check if a row should be kept
        def keep_row(row):
            if max_author_count == -1:
                return True
            work_id_one, work_id_two = row['work_id_one'], row['work_id_two']
            if work_id_one not in encountered_work_ids and work_id_two not in encountered_work_ids:
                encountered_work_ids.add(work_id_one)
                encountered_work_ids.add(work_id_two)
                return True
            return False

        # Apply the filtering
        filtered_df = df_filtered.filter(pl.struct(['work_id_one', 'work_id_two']).map_elements(lambda x: keep_row(x)))

        # Fetch work details
        all_work_ids = set(filtered_df['work_id_one'].to_list() + filtered_df['work_id_two'].to_list())
        work_details = self.fetch_work_details(all_work_ids, works_df, truncated=False, filter_works=True)

        # Prepare data for embedding and similarity calculation
        pairs = list(zip(filtered_df['work_id_one'].to_list(), filtered_df['work_id_two'].to_list()))

        # Initialize dictionaries for different parquet files
        domain_data = {domain: [] for domain in domain_to_field.keys()}
        common_field_data = []
        common_subfield_data = []

        # Process pairs
        for work1_id, work2_id in tqdm(pairs, desc="Processing work pairs"):
            work1 = work_details.get(work1_id, {})
            work2 = work_details.get(work2_id, {})
            if work1 and work2:
                field1 = work1.get('field_string', '')
                field2 = work2.get('field_string', '')
                domain1 = field_to_domain.get(field1, 'No Domain')
                domain2 = field_to_domain.get(field2, 'No Domain')

                if domain1 == domain2:
                    full_string_one = f"{work1.get('title_string', '')} {work1.get('authors_string', '')} {field1} {work1.get('subfield_string', '')}"
                    full_string_two = f"{work2.get('title_string', '')} {work2.get('authors_string', '')} {field2} {work2.get('subfield_string', '')}"

                    pair_data = {
                        'work_id_one': work1_id,
                        'full_string_one': full_string_one,
                        'work_id_two': work2_id,
                        'full_string_two': full_string_two,
                        'common_field': field1 == field2,
                        'common_subfield': work1.get('subfield_string', '') == work2.get('subfield_string', ''),
                    }

                    domain_data[domain1].append(pair_data)

                    if field1 == field2:
                        common_field_data.append(pair_data)
                    if work1.get('subfield_string', '') == work2.get('subfield_string', ''):
                        common_subfield_data.append(pair_data)

        # Function to process domain data
        def process_domain_data(domain, data):
            file_name = f"works_common_authors_{domain.lower().replace(' ', '_')}.parquet"
            file_path = os.path.join(self.datasets_directory, file_name)

            model = self.load_matryoshka_model(self.model_path, self.embedding_dimension)
            model.to(f'cuda:{torch.cuda.current_device()}')

            full_strings = [item['full_string_one'] for item in data] + [item['full_string_two'] for item in data]

            embeddings = []
            batch_size = 64
            for i in range(0, len(full_strings), batch_size):
                batch = full_strings[i:i + batch_size]
                with torch.no_grad():
                    batch_embeddings = model.encode(batch, convert_to_tensor=True).cpu().numpy()
                embeddings.extend(batch_embeddings)

            cos_sims = [1 - cosine(embeddings[i], embeddings[i + len(data)]) for i in range(len(data))]

            for i, item in enumerate(data):
                item['cos_sim'] = cos_sims[i]
                item['embedding_work_one'] = embeddings[i].tolist()
                item['embedding_work_two'] = embeddings[i + len(data)].tolist()

            df = pl.DataFrame(data)
            df.write_parquet(file_path)
            print(f"Saved {file_name}")

        # Process domain data using multiple GPUs
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [executor.submit(process_domain_data, domain, data)
                       for domain, data in domain_data.items() if data]
            concurrent.futures.wait(futures)

        # Save common field and subfield data
        pl.DataFrame(common_field_data).write_parquet(
            os.path.join(self.datasets_directory, "works_common_authors_and_fields.parquet"))
        pl.DataFrame(common_subfield_data).write_parquet(
            os.path.join(self.datasets_directory, "works_common_authors_and_subfields.parquet"))

        print("Restructuring complete.")



    def create_full_string(self, work):
        return f"{work.get('title_string', '')} {work.get('authors_string', '')} {work.get('field_string', '')} {work.get('subfield_string', '')}"

    @measure_time
    def process_and_vectorize_common_elements(self, work_details, pairs):
        common_unigrams, common_bigrams, common_fields, common_subfields = self.process_common_elements(work_details,
                                                                                                        pairs)

        vectorized_unigrams = self.vectorized_common_unigrams(common_unigrams)
        vectorized_bigrams = self.vectorized_common_bigrams(common_bigrams)
        vectorized_fields = self.vectorized_common_fields(common_fields)
        vectorized_subfields = self.vectorized_common_subfields(common_subfields)

        return vectorized_unigrams, vectorized_bigrams, vectorized_fields, vectorized_subfields

    def create_insert_data(self, pairs, work_details, vectorized_unigrams, vectorized_bigrams, vectorized_fields,
                           vectorized_subfields, distances):
        insert_data = []
        for i, (work1_id, work2_id) in enumerate(pairs):
            work1 = work_details.get(work1_id, {})
            work2 = work_details.get(work2_id, {})
            if work1 and work2:
                insert_data.append({
                    'work_id_one': work1_id,
                    'full_string_one': self.create_full_string(work1),
                    'work_id_two': work2_id,
                    'full_string_two': self.create_full_string(work2),
                    'common_unigrams': vectorized_unigrams[i],
                    'common_bigrams': vectorized_bigrams[i],
                    'common_field': bool(vectorized_fields[i]),
                    'common_subfield': bool(vectorized_subfields[i]),
                    'total_score': 0.0,
                    'label': '',
                    'label_int': 0,
                    'p_value': 0.0,
                    'distance': distances[i]  # Add the distance for this pair
                })
        return insert_data

    @measure_time
    def create_sentence_embeddings(self, works_batch_size=100_000, matryoshka_dim=12):
        """
        TODO: We shall parametize the batch size, and matryoshka_dim=12 shall be set by the class arguments.

        :param works_batch_size:
        :param matryoshka_dim:
        :return:
        """


        works_file = self.works_all_collected_file
        df = pl.read_parquet(works_file)

        total_works = len(df)
        total_batches = (total_works + works_batch_size - 1) // works_batch_size

        # Load the Matryoshka model with truncated dimensions
        model = self.load_matryoshka_model(self.model_path, matryoshka_dim)

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


    @measure_time
    def build_vector_index(self, N=20_000_000, use_gpu=True):
        """
        We will be building this on gpus. We must make sure that if we dont have enough memory on a single
        gpu, that we add the vectors to multiple gpu's and train on those instead.


        :param output_directory:
        :param collection_name:
        :param N:
        :param batch_size:
        :param use_gpu:
        :return:
        """
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
    def perform_batch_search(self, index, work_embeddings, k):
        return index.search(work_embeddings, k)

    @measure_time
    def process_common_elements(self, work_details, pairs):
        common_unigrams = []
        common_bigrams = []
        common_fields = []
        common_subfields = []

        for work1_id, work2_id in pairs:
            work1 = work_details.get(work1_id, {})
            work2 = work_details.get(work2_id, {})

            unigrams1 = work1.get('unigrams', [])
            unigrams2 = work2.get('unigrams', [])
            bigrams1 = work1.get('bigrams', [])
            bigrams2 = work2.get('bigrams', [])

            common_unigrams.append(set(unigrams1) & set(unigrams2))
            common_bigrams.append(set(bigrams1) & set(bigrams2))

            # Check if both works have field/subfield before comparing
            field1, field2 = work1.get('field_string'), work2.get('field_string')
            subfield1, subfield2 = work1.get('subfield_string'), work2.get('subfield_string')

            common_fields.append(field1 == field2 if field1 and field2 else None)
            common_subfields.append(subfield1 == subfield2 if subfield1 and subfield2 else None)

        return common_unigrams, common_bigrams, common_fields, common_subfields

    @measure_time
    def vectorized_common_unigrams(self, common_unigrams):
        return [list(unigrams) for unigrams in common_unigrams]

    @measure_time
    def vectorized_common_bigrams(self, common_bigrams):
        return [list(bigrams) for bigrams in common_bigrams]

    @measure_time
    def vectorized_common_fields(self, common_fields):
        return np.array([1 if f is True else 0 if f is False else -1 for f in common_fields], dtype=int)

    @measure_time
    def vectorized_common_subfields(self, common_subfields):
        return np.array([1 if s is True else 0 if s is False else -1 for s in common_subfields], dtype=int)

    @measure_time
    def batch_search_similar_works(self, work_ids, k, index, faiss_to_works_id, distance_threshold=0.1,
                                   print_distance_stats=False):

        valid_work_ids = [work_id for work_id in work_ids if work_id in self.work_details]

        work_embeddings = self.batch_encode_works(
            [self.create_sentence_work(self.work_details[work_id]) for work_id in valid_work_ids])

        # Perform the initial search
        distances, indices = self.perform_batch_search(index, work_embeddings, k)

        # Compute pairwise distances for the retrieved vectors
        pairwise_distances, p_05, p_95 = self.compute_pairwise_distances(work_embeddings, print_stats=print_distance_stats)

        results = []
        for i, work_id in enumerate(valid_work_ids):
            filtered_indices = []
            filtered_distances = []

            for j in range(k):
                works_int_id = int(indices[i][j])  # This is now directly the works_int_id
                if works_int_id not in filtered_indices:  # Check if this ID is already filtered
                    try:
                        similar_work_id = faiss_to_works_id[works_int_id]

                        # Check pairwise distance
                        if filtered_indices and np.min(pairwise_distances[i, filtered_indices]) < distance_threshold:
                            continue  # Skip this result if it's too close to previously added results

                        filtered_indices.append(j)
                        filtered_distances.append(distances[i][j])

                        results.append({
                            'query_work_id': work_id,
                            'similar_work_id': similar_work_id,
                            'distance': float(distances[i][j])
                        })
                    except KeyError:
                        pass
                        # print(f"Warning: No mapping found for works_int_id {works_int_id}")
                        # print(f"Query work_id: {work_id}, j: {j}")

        return pl.DataFrame(results)

    @measure_time
    def compute_pairwise_distances(self, vectors, print_stats=True):
        distances = pdist(vectors)
        distance_matrix = squareform(distances)
        p_05, p_95 = 0.15, 0.75

        if print_stats:
            # Remove zero distances (self-distances)
            non_zero_distances = distances[distances != 0]

            # Calculate statistics
            avg_distance = np.mean(non_zero_distances)
            p_05 = np.percentile(non_zero_distances, 5)
            p_95 = np.percentile(non_zero_distances, 95)

            # Get 10 smallest and 10 largest non-zero distances
            smallest_distances = np.sort(non_zero_distances)[:10]
            largest_distances = np.sort(non_zero_distances)[-10:]

            # Print statistics
            print("\nPairwise Distance Statistics:")
            print(f"Average distance: {avg_distance:.4f}")
            print(f"5th percentile (p-value 0.05): {p_05:.4f}")
            print(f"95th percentile (p-value 0.95): {p_95:.4f}")

            print("\n10 smallest non-zero distances:")
            for i, d in enumerate(smallest_distances, 1):
                print(f"{i}. {d:.4f}")
            print("\n10 largest distances:")
            for i, d in enumerate(largest_distances, 1):
                print(f"{i}. {d:.4f}")

        return distance_matrix, p_05, p_95

    @measure_time
    def fetch_work_details(self, work_ids, works_filtered_df, truncated=False, filter_works=True):
        """
        We have issues with this method.

        :param work_ids:
        :param works_filtered_df:
        :param truncated:
        :param filter_works:
        :return:
        """

        result = {}
        if filter_works:
            df_to_process = works_filtered_df.filter(pl.col('work_id').is_in(work_ids))
        else:
            df_to_process = works_filtered_df

        for row in df_to_process.iter_rows(named=True):
            work_id = row['work_id']
            work_details = {
                'work_id': work_id,
                'field_string': row['field_string'],
                'subfield_string': row['subfield_string'],
                'title_string': row['title_string'],
                'authors_string': row['authors_string'],
            }
            if not truncated:
                work_details.update({
                    'unigrams': row['unigrams'],
                    'bigrams': row['bigrams'],
                })
            result[work_id] = work_details

        return result

    @measure_time
    def get_gram_scores(self, grams, gram_df):
        """
        Get gram scores from the gram DataFrame using Polars.

        :param grams: Polars Series of grams
        :param gram_df: Polars DataFrame with gram scores
        :return: Dictionary of gram scores
        """
        gram_type = "unigram_type" if 'unigram_type' in gram_df.columns else "bigram_type"

        # Filter the gram_df for the grams we need
        filtered_df = gram_df.filter(pl.col(gram_type).is_in(grams))

        # Convert to dictionary
        scores = dict(zip(filtered_df[gram_type], filtered_df['score']))

        # Return dictionary with default value of 2.5 for missing grams
        return {gram: float(scores.get(gram, 2.5)) for gram in grams}

    @measure_time
    def batch_encode_works(self, work_strings, batch_size=32):
        return self.model.encode(work_strings, batch_size=batch_size)

    def reconstruct_abstract(self, abstract_inverted_index):
        if not abstract_inverted_index:
            return ""

        # Get the maximum position
        max_position = max(max(positions) for positions in abstract_inverted_index.values())

        # Create a list to hold words in their positions
        words = [''] * (max_position + 1)

        # Place each word in its correct position(s)
        for word, positions in abstract_inverted_index.items():
            for position in positions:
                words[position] = word

        # Join the words to form the abstract
        return ' '.join(words).strip()


    @measure_time
    def remove_single_count_items(self, counter, min_count=1):
        return Counter({item: count for item, count in counter.items() if count > min_count})

    @measure_time
    def save_ngram_data(self, ngram_counts, ngram_type, output_dir):
        ngram_data = [
            (gram, count, 0.0) for gram, count in ngram_counts.items()
        ]
        ngram_df = pl.DataFrame(ngram_data, schema=[f'{ngram_type}_type', 'count', 'score'])

        file_path = os.path.join(output_dir, f'{ngram_type}_data.parquet')
        ngram_df.write_parquet(file_path)

        print(f"{ngram_type.capitalize()} data saved to {file_path}. Total rows: {len(ngram_df)}")



    @measure_time
    def generate_all_work_id_pairs_dataset(self, sort_by_distance=True):
        """
        TODO: sort_by_distance is for curriculum learning. It is probably a good idea to be wary of using
            this as it can concentrate particular kinds of data toward the end.

        Consider this method.
            We are interesting in creating a method here that generates a triplets dataset, but uses a particular method to filter out pairs if we find too many.
            Since we have distances, I would like to generate a particular system. When we try to merge two pairs into triplets, we pick the first one that comes along, it seems.
            I generally do not like this strategy, as it does not allow us to pick the best option.
            I kinda want ones that have their triplets so that the anchor is close to the positive, and the positive is close to the negative,
            but the anchor is reasonably far away from the positive. I also want candidates have fairly high total_scores. So, we have to think about this.

            One option is that we generate encodings of each string, and then compare the similarities of anchor, positive, anchor negative, and positive negative,
            for the anchor, positive, and all negative candidates.
            Then we make sure each candidate is below a certain threshold of similarity, and pick the highest similarity of any of the negative candidates to the positive.

            Another is to filter out any pairs that have a total_score in the top 1% percentile. This will likely include a lot of augmentations. We could do that in the augmented method actually.


        :param sort_by_distance:
        :return:
        """

        print("Generating all work ID pairs dataset...")


        files = [
            self.works_augmented_data_file
        ]

        all_pairs = []
        work_id_counts = {}

        # Load all pairs and calculate median score
        for file in files:
            if os.path.exists(file):
                df = pl.read_parquet(file)
                df = df.with_columns(pl.lit(os.path.basename(file).replace('.parquet', '')).alias('source'))
                df = df.with_columns(pl.col('augmentation_type').fill_null(pl.Null))

                # Select only the required columns
                selected_df = df.select(['work_id_one', 'work_id_two', 'total_score', 'source', 'augmentation_type'])
                all_pairs.extend(selected_df.to_numpy().tolist())

            gc.collect()

        # Shuffle all_pairs to mix up the sources
        random.shuffle(all_pairs)

        # Calculate median score
        scores = [pair[2] for pair in all_pairs]
        mean_score = np.mean(scores)

        # Second pass: Populate work_id_counts
        for pair in all_pairs:
            work_id_one, work_id_two, total_score = pair[0], pair[1], pair[2]

            if work_id_one not in work_id_counts:
                work_id_counts[work_id_one] = {'positive': 0, 'negative': 0, 'total': 0}
            if work_id_two not in work_id_counts:
                work_id_counts[work_id_two] = {'positive': 0, 'negative': 0, 'total': 0}

            if total_score >= mean_score:
                work_id_counts[work_id_one]['positive'] += 1
                work_id_counts[work_id_two]['positive'] += 1
            else:
                work_id_counts[work_id_one]['negative'] += 1
                work_id_counts[work_id_two]['negative'] += 1

            work_id_counts[work_id_one]['total'] += 1
            work_id_counts[work_id_two]['total'] += 1

        print("length all_pairs: ", len(all_pairs))

        gc.collect()

        # Filter pairs
        filtered_pairs = []
        work_id_occurrences = {}

        for pair in all_pairs:
            work_id_one, work_id_two = pair[0], pair[1]

            # Check if at least one of the work_ids has both positive and negative examples or
            # one has a positive and the other has a negative
            if (
                    ((work_id_counts[work_id_one]['positive'] > 0 and work_id_counts[work_id_one]['negative'] >= 0) and
                     (work_id_counts[work_id_two]['positive'] >= 0 and work_id_counts[work_id_two]['negative'] > 0)) or

                    ((work_id_counts[work_id_one]['positive'] >= 0 and work_id_counts[work_id_one]['negative'] > 0) and
                     (work_id_counts[work_id_two]['positive'] > 0 and work_id_counts[work_id_two]['negative'] >= 0)) or

                    ((work_id_counts[work_id_one]['positive'] > 0 and work_id_counts[work_id_one]['negative'] > 0) or
                     (work_id_counts[work_id_two]['positive'] > 0 and work_id_counts[work_id_two]['negative'] > 0))
            ):

                # Increase occurrence limit to 3
                if (work_id_occurrences.get(work_id_one, {'one': 0, 'two': 0})['one'] < 4 and
                        work_id_occurrences.get(work_id_two, {'one': 0, 'two': 0})['two'] < 4):

                    # Add the pair to filtered_pairs
                    filtered_pairs.append(pair)

                    # Update occurrence counts
                    if work_id_one not in work_id_occurrences:
                        work_id_occurrences[work_id_one] = {'one': 0, 'two': 0}
                    work_id_occurrences[work_id_one]['one'] += 1

                    if work_id_two not in work_id_occurrences:
                        work_id_occurrences[work_id_two] = {'one': 0, 'two': 0}
                    work_id_occurrences[work_id_two]['two'] += 1

        print("Length of all_pairs: ", len(all_pairs))
        print("Length of filtered_pairs: ", len(filtered_pairs))
        gc.collect()

        # Convert to DataFrame
        columns = ['work_id_one', 'work_id_two', 'total_score', 'source', 'augmentation_type']
        filtered_pairs_df = pl.DataFrame(filtered_pairs, schema=columns)

        # Calculate z-scores and normalize
        mean_score = filtered_pairs_df['total_score'].mean()
        std_score = filtered_pairs_df['total_score'].std()
        filtered_pairs_df = filtered_pairs_df.with_columns([
            ((pl.col('total_score') - mean_score) / std_score).alias('z_score'),
        ])
        filtered_pairs_df = filtered_pairs_df.with_columns([
            self.sigmoid_normalize(pl.col('z_score')).alias('normalized_z_score')
        ])

        output_file = os.path.join(self.datasets_directory, f"triplet_ids.parquet")
        filtered_pairs_df.write_parquet(output_file)
        print(f"Saved triplet_work_ids_only to {output_file}")

        # Generate triplets
        positive_pairs = filtered_pairs_df.filter(pl.col('normalized_z_score') >= 0)
        negative_pairs = filtered_pairs_df.filter(pl.col('normalized_z_score') < 0)

        triplets = positive_pairs.join(negative_pairs, on='work_id_one', how='inner', suffix='_neg')
        triplets = triplets.filter(pl.col('work_id_two') != pl.col('work_id_two_neg'))
        triplets = triplets.drop(['z_score', 'z_score_neg'])

        # Calculate max_pos_neg_distance
        triplets = triplets.with_columns([
            (pl.col('normalized_z_score') - pl.col('normalized_z_score_neg')).alias('max_pos_neg_distance')
        ])

        # Sort by max_pos_neg_distance if requested
        if sort_by_distance:
            triplets = triplets.sort('max_pos_neg_distance', descending=True)

        # Rename columns for final output
        triplets = triplets.rename({
            'work_id_one': 'anchor',
            'work_id_two': 'positive',
            'work_id_two_neg': 'negative',
            'normalized_z_score': 'z_score_pos',
            'normalized_z_score_neg': 'z_score_neg',
            'total_score': 'total_score_pos',
            'total_score_neg': 'total_score_neg',
            'source': 'source_pos',
            'source_neg': 'source_neg',
            'augmentation_type': 'augmentation_type_pos',
            'augmentation_type_neg': 'augmentation_type_neg'
        })

        # Fetch work details and create full strings
        works_filtered_df = pl.read_parquet(self.works_all_collected_file)

        all_work_ids = set(
            triplets['anchor'].to_list() + triplets['positive'].to_list() + triplets['negative'].to_list())
        work_details = self.fetch_work_details(all_work_ids, works_filtered_df, truncated=True)

        # Create full strings
        triplets = triplets.with_columns([
            pl.col('anchor').map_elements(lambda x: self.create_full_string(work_details.get(x, {}))).alias(
                'anchor_string'),
            pl.col('positive').map_elements(lambda x: self.create_full_string(work_details.get(x, {}))).alias(
                'positive_string'),
            pl.col('negative').map_elements(lambda x: self.create_full_string(work_details.get(x, {}))).alias(
                'negative_string')
        ])

        # Select and order final columns
        final_columns = [
            'anchor', 'positive', 'negative',
            'anchor_string', 'positive_string', 'negative_string',
            'total_score_pos', 'total_score_neg',
            'z_score_pos', 'z_score_neg',
            'max_pos_neg_distance',
            'source_pos', 'source_neg',
            'augmentation_type_pos', 'augmentation_type_neg'
        ]

        triplets = triplets.select(final_columns)

        # Initialize dictionaries to keep track of occurrences
        anchor_occurrences = {}
        positive_occurrences = {}
        negative_occurrences = {}

        # List to store the filtered triplets
        filtered_triplets = []

        for row in triplets.iter_rows(named=True):
            anchor = row['anchor']
            positive = row['positive']
            negative = row['negative']

            # Check anchor occurrence
            if anchor not in anchor_occurrences:
                # Check positive occurrence
                if positive_occurrences.get(positive, 0) < 2:
                    # Check negative occurrence
                    if negative_occurrences.get(negative, 0) < 2:
                        # If all conditions are met, add the triplet
                        filtered_triplets.append(row)

                        # Update occurrences
                        anchor_occurrences[anchor] = 1
                        positive_occurrences[positive] = positive_occurrences.get(positive, 0) + 1
                        negative_occurrences[negative] = negative_occurrences.get(negative, 0) + 1

        # Convert the filtered triplets back to a DataFrame
        triplets = pl.DataFrame(filtered_triplets)

        # Print head and tail of the DataFrame
        print("\nHead of the DataFrame (20 rows):")
        print(triplets.head(20))
        print("\nTail of the DataFrame (20 rows):")
        print(triplets.tail(20))
        print("length of triplets: ", len(triplets))

        print("Original number of triplets:", len(triplets))
        print("Number of triplets after filtering:", len(filtered_triplets))

        # Print column names after final selection
        print("\nFinal column names:")
        print(triplets.columns)

        triplets.write_parquet(self.triplets_file)
        print(f"\nSaved triplets to {self.triplets_file}")

    def sigmoid_normalize(self, x):
        return (2 / (1 + pl.exp(-x))) - 1

    @measure_time
    def triplets_quality_control_statistics(self):
        print("Performing quality control statistics on triplets...")

        # Load the triplets parquet file
        triplets_file = os.path.join(self.datasets_directory, "triplets.parquet")
        if not os.path.exists(triplets_file):
            print(f"Error: Triplets file not found at {triplets_file}")
            return

        triplets_df = pl.read_parquet(triplets_file)

        # Count occurrences of each work_id
        all_work_ids = pl.concat([
            triplets_df.select('anchor'),
            triplets_df.select('positive'),
            triplets_df.select('negative')
        ])
        work_id_counts = all_work_ids.value_counts()

        # Count work_ids appearing twice in the same row
        same_row_duplicates = triplets_df.filter(
            (pl.col('anchor') == pl.col('positive')) |
            (pl.col('anchor') == pl.col('negative')) |
            (pl.col('positive') == pl.col('negative'))
        ).shape[0]

        # Count work_ids appearing different number of times
        appear_once = work_id_counts.filter(pl.col('counts') == 1).shape[0]
        appear_twice = work_id_counts.filter(pl.col('counts') == 2).shape[0]
        appear_thrice = work_id_counts.filter(pl.col('counts') == 3).shape[0]
        appear_four_or_more = work_id_counts.filter(pl.col('counts') >= 4).shape[0]

        # Print statistics
        print(f"\nTotal number of triplets: {triplets_df.shape[0]}")
        print(f"Total unique work_ids: {work_id_counts.shape[0]}")
        print(f"\nWork_ids appearing twice in the same row: {same_row_duplicates}")
        print(f"Work_ids appearing only once in the dataset: {appear_once}")
        print(f"Work_ids appearing twice in the dataset: {appear_twice}")
        print(f"Work_ids appearing three times in the dataset: {appear_thrice}")
        print(f"Work_ids appearing four or more times in the dataset: {appear_four_or_more}")

        # Additional statistics
        print(f"\nMaximum appearances of a single work_id: {work_id_counts['counts'].max()}")
        print("\nDistribution of work_id appearances:")
        appearance_distribution = work_id_counts.group_by('counts').agg(pl.count()).sort('counts')
        for row in appearance_distribution.iter_rows(named=True):
            print(f"  {row['counts']} time(s): {row['count']} work_ids")

        # Check for any missing values
        missing_values = triplets_df.null_count()
        if missing_values.sum().sum() > 0:
            print("\nWarning: Missing values detected:")
            print(missing_values.filter(pl.all().is_not_null()))
        else:
            print("\nNo missing values detected.")

        # Check for duplicate triplets
        duplicate_triplets = triplets_df.is_duplicated().sum()
        print(f"\nNumber of duplicate triplets: {duplicate_triplets}")

        # Summary of total_score and z_score distributions
        print("\nSummary statistics for scores:")
        score_summary = triplets_df.select(
            ['total_score_pos', 'total_score_neg', 'z_score_pos', 'z_score_neg']).describe()
        print(score_summary)

        print("\nQuality control statistics completed.")

    def create_fine_tuning_datasets(self, triplets_file, output_directory):
        """
        We may modify this to also create fine tuning scores based off p_values or rather z_scores,
        just for particular curriculum learning datasets.

        :param triplets_file:
        :param output_directory:
        :return:
        """

        # Read the triplets file
        triplets_df = pl.read_parquet(triplets_file)

        # Define filters for authors and titles datasets
        author_augmentations = [
            'author_field', 'all_authors_field', 'one_author_field_subfield',
            'two_authors_field_subfield', 'two_authors_field', 'all_authors_field_subfield'
        ]
        title_augmentations = ['full_title', 'full_title_field', 'full_title_field_subfield']

        # Filter for authors dataset
        authors_df = triplets_df.filter(
            (pl.col('augmentation_type_pos').is_in(author_augmentations) |
             pl.col('augmentation_type_neg').is_in(author_augmentations) |
             pl.col('source_pos').is_in(['works_common_authors', 'works_augmented_data']) |
             pl.col('source_neg').is_in(['works_common_authors', 'works_augmented_data']))
        )

        # Filter for titles dataset
        titles_df = triplets_df.filter(
            (pl.col('augmentation_type_pos').is_in(title_augmentations) |
             pl.col('augmentation_type_neg').is_in(title_augmentations) |
             pl.col('source_pos') == 'common_title_works' |
             pl.col('source_neg') == 'common_title_works')
        )

        # Save the filtered datasets
        authors_file = os.path.join(output_directory, 'fine_tuning_authors.parquet')
        titles_file = os.path.join(output_directory, 'fine_tuning_titles.parquet')

        authors_df.write_parquet(authors_file)
        titles_df.write_parquet(titles_file)

        print(f"Authors fine-tuning dataset saved to: {authors_file}")
        print(f"Titles fine-tuning dataset saved to: {titles_file}")

        # Print statistics
        print(f"\nAuthors dataset size: {len(authors_df)} triplets")
        print(f"Titles dataset size: {len(titles_df)} triplets")

        # Print sample rows from each dataset
        print("\nSample rows from Authors dataset:")
        print(authors_df.head(5))
        print("\nSample rows from Titles dataset:")
        print(titles_df.head(5))


def setup_directories(environment='local'):
    # TODO: make sure the directory works for linux env.
    # TODO: we now are going to start using the best_model here: "E:\HugeDatasetBackup\cloud_models\matryoshka_model\best_model"

    if environment == 'local':
        if platform.system() == 'Windows':
            base_dir = r"E:\HugeDatasetBackup"
        else:
            base_dir = os.path.expanduser("~/HugeDatasetBackup")
    elif environment == 'cloud':
        base_dir = r"/workspace"
    else:
        raise ValueError("Invalid environment. Choose 'local' or 'cloud'.")

    directories = {
        'base': base_dir,
        'model': os.path.join(base_dir, 'DATA_CITATION_GRABBER', 'models', 'best_model'),
        'output': os.path.join(base_dir, 'cloud_models'),
        'datasets': os.path.join(base_dir, 'cloud_datasets'),
        'embeddings': os.path.join(base_dir, 'cloud_embeddings'),
        'vectordbs': os.path.join(base_dir, 'cloud_vectordbs'),
        'ngrams': os.path.join(base_dir, 'cloud_ngrams'),
    }

    for dir_name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    return directories

if __name__ == "__main__":
    # TODO: we will want to use this

    # Choose 'local' or 'cloud' based on your environment
    env = 'local'
    dirs = setup_directories(env)

    run_params = {
        'load_and_print_data': False,
        'create_works_notopic_all': True,
        'collect_all_works_metadata': True,
        'restructure_common_authors': True,
        'restructure_augmented_data': True,
        'create_sentence_embeddings': True,
        'build_vector_index': True,
        'generate_training_pairs': True,
        'generate_all_work_id_pairs_dataset': False,
    }

    encoder = CloudDatasetConstructionSentenceEncoderT1(
        model_path=dirs['model'],
        output_directory=dirs['output'],
        datasets_directory=dirs['datasets'],
        embeddings_directory=dirs['embeddings'],
        ngrams_directory=dirs['ngrams'],
        vectordb_directory=dirs['vectordbs'],
        run_params=run_params,
        num_knn_pairs=100_000,
        num_works_collected=100_000,
        mongo_url="mongodb://localhost:27017/",
        mongo_database_name="OpenAlex",
        mongo_works_collection_name="Works"
    )

    encoder.run()
    # encoder.triplets_quality_control_statistics()


