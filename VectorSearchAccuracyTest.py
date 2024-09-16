import csv
import datetime
import json
import math
import os
import time
from collections import defaultdict
from typing import List
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

# from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer
# from sentence_transformers.training_args import SentenceTransformerTrainingArguments


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper


class VectorSearchAccuracyTest:
    """
    TODO: We are supposed to build our index only from the encodings of the full_strings, and not the varied query_strings.
        Those are to be used for search run_tests and run_tests_second instead.

    """


    def __init__(self, previous_model_path, output_directory):
        # Initialize MongoDB connection
        self.mongo_url = "mongodb://localhost:27017/"
        self.mongo_database_name = "OpenAlex"
        self.mongo_works_collection_name = "Works"
        self.mongo_client = MongoClient(self.mongo_url)
        self.mongo_db = self.mongo_client[self.mongo_database_name]
        self.works_collection = self.mongo_db[self.mongo_works_collection_name]

        self.previous_model_path = previous_model_path
        self.output_directory = output_directory

        # PostgreSQL database information
        self.pg_host = "localhost"
        self.pg_database = "CitationData"
        self.pg_user = "postgres"
        self.pg_password = "Cl0venh00f$$"
        self.pg_port = 5432
        # "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\models\\models--sentence-transformers--all-MiniLM-L6-v2\\snapshots\\8b3219a92973c328a8e22fadcfa821b5dc75636a"
        # r"C:\Users\doren\.cache\huggingface\hub\models--Snowflake--snowflake-arctic-embed-xs\snapshots\55416e45af748c8b882e5b8d3e202ab4713f6110"
        # Load models

        self.original_model = SentenceTransformer(r"C:\Users\doren\.cache\huggingface\hub\models--Snowflake--snowflake-arctic-embed-xs\snapshots\55416e45af748c8b882e5b8d3e202ab4713f6110")
        self.fine_tuned_model = SentenceTransformer(previous_model_path)

        self.original_index_path = os.path.join(self.output_directory, "index_original.bin")
        self.original_mapping_path = os.path.join(self.output_directory, "id_mapping_original.arrow")
        self.original_index_info_path = os.path.join(self.output_directory, "original_index_info.json")

        self.result_counter = 0
        self.model_config = {
            "original_model": r"C:\Users\doren\.cache\huggingface\hub\models--Snowflake--snowflake-arctic-embed-xs\snapshots\55416e45af748c8b882e5b8d3e202ab4713f6110",
            "fine_tuned_model": previous_model_path
        }

        # print("Original model:", self.original_model)
        print("Fine-tuned model:", self.fine_tuned_model)

    def extract_work_info(self, work_data):
        display_name = work_data.get("display_name", "")

        author_names = [authorship.get("author", {}).get("display_name", "") for authorship in
                        work_data.get("authorships", [])]
        author_names = author_names[:20]

        topics = work_data.get("topics", [])

        field_names = list(
            set(field.get("display_name", "") for topic in topics for field in [topic.get("field", {})] if field))
        subfield_names = list(set(
            subfield.get("display_name", "") for topic in topics for subfield in [topic.get("subfield", {})] if
            subfield))

        # full_string = str(display_name + " " + author_names)
        # if len(full_string) < 8:
        #     return None

        return {
            "work_id": work_data.get("id"),
            "works_int_id": work_data.get("works_int_id"),
            "display_name": display_name,
            "author_names": author_names,
            "field_names": field_names,
            "subfield_names": subfield_names,
        }

    def create_queries(self, work_info, full_query_only=False):
        def clean_name(name):
            if name:
                return ' '.join(name.strip().split())  # Remove extra spaces
            return ""

        display_name = clean_name(work_info["display_name"])
        author_names = [clean_name(name) for name in work_info["author_names"] if name.strip()]
        author_names = author_names[:20]
        field_names = [clean_name(name) for name in work_info["field_names"] if name.strip()]
        subfield_names = [clean_name(name) for name in work_info["subfield_names"] if name.strip()]

        title_words = work_info.get('display_name', ' ')
        if title_words:
            words = title_words.split()
            trigram = " ".join(words[:min(3, len(words))]) + " "
        else:
            trigram = " "

        full_query = f"{display_name} {' '.join(author_names)} {' '.join(field_names)} {' '.join(subfield_names)}"

        if full_query_only:
            return [full_query], full_query

        queries = [
            f"{display_name} {' '.join(field_names)} ",
            f"{trigram} {' '.join(field_names)} {' '.join(subfield_names)} ",
            f"{trigram} {' '.join(field_names)} ",
            f"{' '.join(author_names[:2])} {' '.join(field_names)}",
            f"{author_names[0] if author_names else ''} {' '.join(field_names)} {' '.join(subfield_names)} ",
            f"{author_names[0] if author_names else ' '} {' '.join(field_names)}",
            full_query
        ]

        # Remove any empty queries
        queries = [q.strip() for q in queries if q.strip()]

        # Ensure consistent number of queries
        while len(queries) < 8:
            queries.append(" ")

        queries = queries[:8]  # Limit to 8 queries

        return queries, full_query

    def create_embeddings_and_index(self, num_works=500):
        last_work_int_id = 1500000
        query = {"works_int_id": {"$gt": last_work_int_id}}

        work_infos = []
        works_cursor = self.works_collection.find(query).sort("works_int_id", 1)

        for work in works_cursor:
            work_info = self.extract_work_info(work)
            if work_info is not None:  # Only add work_info if it's not None
                work_infos.append(work_info)
                if len(work_infos) >= num_works:
                    break

        original_embeddings = []
        fine_tuned_embeddings = []
        for i, work_info in enumerate(work_infos):
            if i % 1000 == 0:
                print("embedding: ", i)
            [full_query], _ = self.create_queries(work_info, full_query_only=True)
            original_embeddings.append(self.original_model.encode(full_query))
            fine_tuned_embeddings.append(self.fine_tuned_model.encode(full_query))

        self.build_vector_index(original_embeddings, work_infos, "original")
        self.build_vector_index(fine_tuned_embeddings, work_infos, "fine_tuned")

        return work_infos

    def can_reuse_original_index(self, num_works):
        if not os.path.exists(self.original_index_info_path):
            return False

        with open(self.original_index_info_path, 'r') as f:
            info = json.load(f)

        return (info['model_path'] == self.original_model.get_model_path() and
                info['num_works'] == num_works and
                os.path.exists(self.original_index_path) and
                os.path.exists(self.original_mapping_path))

    def save_original_index_info(self, num_works):
        info = {
            'model_path': self.original_model.get_model_path(),
            'num_works': num_works
        }
        with open(self.original_index_info_path, 'w') as f:
            json.dump(info, f)

    def get_work_infos(self, num_works):
        last_work_int_id = 1500000
        query = {"works_int_id": {"$gt": last_work_int_id}}

        work_infos = []
        works_cursor = self.works_collection.find(query).sort("works_int_id", 1).limit(num_works)

        for work in works_cursor:
            work_info = self.extract_work_info(work)
            if work_info is not None:
                work_infos.append(work_info)

        return work_infos



    @measure_time
    def build_vector_index(self, embeddings, work_infos, collection_name, N=20000000):
        print(f"Building vector index for {collection_name}...")

        # Create GPU resources

        # Extract data for indexing
        faiss_ids = list(range(len(work_infos[:N])))
        work_int_ids = [work_info['works_int_id'] for work_info in work_infos[:N]]
        item_ids = [work_info['work_id'] for work_info in work_infos[:N]]
        embeddings = np.array(embeddings[:N])

        # Create and train the index
        index = self.create_faiss_index(embeddings, np.array(faiss_ids), item_ids, collection_name)

        # Save the index
        index_path = os.path.join(self.output_directory, f"index_{collection_name.lower()}.bin")
        faiss.write_index(index, index_path)

        # Create and save the mapping DataFrame
        df = pd.DataFrame({
            'faiss_id': faiss_ids,
            'works_int_id': work_int_ids,
            f'{collection_name}_id': item_ids,
        })
        mapping_path = os.path.join(self.output_directory, f"id_mapping_{collection_name.lower()}.arrow")
        feather.write_feather(df, mapping_path)

        print("mapping path saved: ", mapping_path)
        print("index_path saved: ", index_path)

        print(f"Vector index building completed for {collection_name}.")

        return index_path, mapping_path


    @measure_time
    def calculate_index_parameters(self, collection_size):
        if collection_size < 10_000:
            nlist = int(800)
            return f"IVF{nlist}", nlist, None
        if 10_000 <= collection_size < 30_000:
            nlist = 32 * int(math.sqrt(collection_size))
            return f"IVF{nlist}", nlist, None
        elif 30_000 <= collection_size < 100_000:
            nlist = 48 * int(math.sqrt(collection_size))
            return f"IVF{nlist}", nlist, None
        elif 100_000 <= collection_size < 1_000_000:
            nlist = 64 * int(math.sqrt(collection_size))
            return f"IVF{nlist}", nlist, None
        elif 1_000_000 <= collection_size < 10_000_000:
            return "IVF65536_HNSW32", 65536, 32
        elif 10_000_000 <= collection_size < 100_000_000:
            return "IVF262144_HNSW32", 262144, 32
        else:  # 100M or more
            return "IVF1048576_HNSW32", 262144, 32

    @measure_time
    def create_faiss_index(self, embeddings, int_ids, item_ids, collection_name):
        """
        Create a FAISS index with the given parameters. Execution time of create_faiss_index: 6963.895109 seconds

        :param embeddings: numpy array of embeddings
        :param int_ids: list of integer IDs
        :param item_ids: list of item IDs
        :param collection_name: name of the collection
        :return: FAISS index
        """
        d = embeddings.shape[1]
        collection_size = len(int_ids)
        index_type, nlist, hnsw_m = self.calculate_index_parameters(collection_size)

        if "HNSW" in index_type:
            quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, 32, 8)
        else:
            index = faiss.index_factory(d, index_type + ",PQ32")

        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(512, nlist // 4)  # Adjust nprobe based on nlist

        print("index.nprobe", index.nprobe)

        print("nlist // 4: ", nlist // 4)

        return index


    def encode_queries(self, queries: List[str], model: SentenceTransformer):
        return model.encode(queries)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


    def get_random_works(self, n=1000):
        return list(self.works_collection.aggregate([{"$sample": {"size": n}}]))

    def calculate_precision_at_k(self, relevant_items, retrieved_items, k):
        """Calculate Precision@k"""
        return len(set(relevant_items) & set(retrieved_items[:k])) / k

    def calculate_recall_at_k(self, relevant_items, retrieved_items, k):
        """Calculate Recall@k"""
        return len(set(relevant_items) & set(retrieved_items[:k])) / len(relevant_items)

    def calculate_average_precision(self, relevant_items, retrieved_items):
        """Calculate Average Precision"""
        precisions = []
        relevant_count = 0
        for i, item in enumerate(retrieved_items):
            if item in relevant_items:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        return sum(precisions) / len(relevant_items) if precisions else 0

    @measure_time
    def evaluate_search(self, work_infos, num_test_works=100, k_values=[5, 10, 20, 50]):
        test_works = work_infos[:num_test_works]
        query_types = [
            "Full title + field: ",
            "Trigram from title + field + subfield: ",
            "Trigram from title + field: ",
            "First 2 authors + field ",
            "One author + field + subfield: ",
            "Author + field: ",
        ]

        metrics = {
            'original': defaultdict(lambda: defaultdict(list)),
            'fine_tuned': defaultdict(lambda: defaultdict(list))
        }

        original_index = faiss.read_index(os.path.join(self.output_directory, "index_original.bin"))
        fine_tuned_index = faiss.read_index(os.path.join(self.output_directory, "index_fine_tuned.bin"))

        # Load id mappings
        original_mapping = pd.read_feather(os.path.join(self.output_directory, "id_mapping_original.arrow"))
        fine_tuned_mapping = pd.read_feather(os.path.join(self.output_directory, "id_mapping_fine_tuned.arrow"))

        max_k = max(k_values)

        for model_name, model, index, mapping in [
            ('original', self.original_model, original_index, original_mapping),
            ('fine_tuned', self.fine_tuned_model, fine_tuned_index, fine_tuned_mapping)
        ]:
            for query_type in query_types:
                for work in tqdm(test_works, desc=f"Evaluating {model_name} - {query_type}"):
                    try:

                        queries, full_query = self.create_queries(work)
                        work_int_id = work['works_int_id']

                        query_embeddings = model.encode([queries[query_types.index(query_type)]])
                        _, indices = index.search(query_embeddings, max_k)

                        retrieved_faiss_ids = indices[0].tolist()
                        retrieved_work_int_ids = mapping.loc[
                            mapping['faiss_id'].isin(retrieved_faiss_ids), 'works_int_id'
                        ].tolist()

                        for k in k_values:
                            recall = self.calculate_recall_at_k([work_int_id], retrieved_work_int_ids, k)
                            metrics[model_name][query_type][k].append(recall)

                    except Exception as e:
                        print("Error: ", e)

        return metrics

    @measure_time
    def evaluate_search_domain(self, num_test_works=500, k_values=[5, 10, 20, 50]):
        input_file = r"D:\evaluate_search_vectordb\work_data.parquet"
        df = pd.read_parquet(input_file)

        # Print data types of columns
        print("DataFrame column types:")
        print(df.dtypes)

        # Print a few rows to check the data
        print("\nFirst few rows of the DataFrame:")
        print(df.head())
        print(df.head(20).to_string())

        print("\nFirst last rows of the DataFrame:")
        print(df.tail(20).to_string())

        domains = df["domain"].unique()
        metrics = {
            'original': {domain: defaultdict(lambda: defaultdict(list)) for domain in domains},
            'fine_tuned': {domain: defaultdict(lambda: defaultdict(list)) for domain in domains}
        }

        original_index = faiss.read_index(os.path.join(self.output_directory, "index_original.bin"))
        fine_tuned_index = faiss.read_index(os.path.join(self.output_directory, "index_fine_tuned.bin"))

        original_mapping = pd.read_feather(os.path.join(self.output_directory, "id_mapping_original.arrow"))
        fine_tuned_mapping = pd.read_feather(os.path.join(self.output_directory, "id_mapping_fine_tuned.arrow"))

        max_k = max(k_values)

        for domain in domains:
            domain_df = df[df["domain"] == domain].sample(n=min(num_test_works, len(df[df["domain"] == domain])))

            for model_name, model, index, mapping in [
                ('original', self.original_model, original_index, original_mapping),
                ('fine_tuned', self.fine_tuned_model, fine_tuned_index, fine_tuned_mapping)
            ]:
                for _, row in tqdm(domain_df.iterrows(), desc=f"Evaluating {model_name} - {domain}",
                                   total=len(domain_df)):
                    try:
                        query_embedding = model.encode([str(row["work_string"])])
                        _, indices = index.search(query_embedding, max_k)

                        retrieved_faiss_ids = indices[0].tolist()
                        retrieved_work_int_ids = mapping.loc[
                            mapping['faiss_id'].isin(retrieved_faiss_ids), 'works_int_id'
                        ].tolist()

                        for k in k_values:
                            recall = self.calculate_recall_at_k([row["works_int_id"]], retrieved_work_int_ids, k)
                            metrics[model_name][domain]["recall"][k].append(recall)
                    except Exception as e:
                        print(f"Error processing row: {row}")
                        print(f"Error message: {str(e)}")

        return metrics

    # def print_evaluation_results_domain(self, metrics, k_values=[5, 10, 20, 50]):
    #     domains = list(metrics['original'].keys())
    #
    #     for domain in domains:
    #         print(f"\nResults for domain: {domain}")
    #         header = f"{'Model':<15}"
    #         for k in k_values:
    #             header += f"{'Recall@' + str(k):<12}"
    #         print(header)
    #         print("-" * (15 + 12 * len(k_values)))
    #
    #         for model_name in ['original', 'fine_tuned']:
    #             row = f"{model_name:<15}"
    #             for k in k_values:
    #                 recall = np.mean(metrics[model_name][domain]["recall"][k])
    #                 row += f"{recall:.3f}{' ' * 8}"
    #             print(row)
    #
    #     print("\nOverall Results:")
    #     header = f"{'Model':<15}"
    #     for k in k_values:
    #         header += f"{'Avg Recall@' + str(k):<16}"
    #     print(header)
    #     print("-" * (15 + 16 * len(k_values)))
    #
    #     results = {
    #         "query_type": [],
    #         "k": [],
    #         "original_recall": [],
    #         "fine_tuned_recall": [],
    #         "ratio": []
    #     }
    #
    #     for model_name in ['original', 'fine_tuned']:
    #         row = f"{model_name:<15}"
    #         for k in k_values:
    #             avg_recall = np.mean([np.mean(metrics[model_name][domain]["recall"][k]) for domain in domains])
    #             row += f"{avg_recall:.3f}{' ' * 12}"
    #
    #             results["query_type"].append(f"{model_name} - {k}")
    #             results["k"].append(k)
    #             results[f"{model_name}_recall"].append(avg_recall)
    #
    #         print(row)
    #
    #     # Calculate ratios
    #     for i in range(len(results["k"])):
    #         if results["fine_tuned_recall"][i] != 0:
    #             ratio = results["original_recall"][i] / results["fine_tuned_recall"][i]
    #         else:
    #             ratio = float('inf')
    #         results["ratio"].append(ratio)
    #
    #     print("\nRatios (Original Recall : Fine-tuned Recall):")
    #     for i, k in enumerate(k_values):
    #         print(f"Recall@{k}: {results['ratio'][i]:.3f}:1")
    #
    #     return results

    def print_evaluation_results(self, metrics, k_values=[5, 10, 20, 50]):
        k_values = [5, 10, 20, 50]
        header = f"{'Query Type':<40}"
        for k in k_values:
            header += f"{'Original Recall@ ' + str(k):<12} | {'Fine Recall@ ' + str(k):<12}  ## "
        print(header)
        print("-" * (38 + 48 * len(k_values)))

        avg_original = {k: [] for k in k_values}
        avg_fine_tuned = {k: [] for k in k_values}

        for query_type in metrics['original'].keys():
            row = f"{query_type:<40}"
            for k in k_values:
                orig_recall = np.mean(metrics['original'][query_type][k])
                ft_recall = np.mean(metrics['fine_tuned'][query_type][k])
                row += f"{orig_recall:.3f}{' ' * 14}{ft_recall:.3f}{' ' * 14}"
                avg_original[k].append(orig_recall)
                avg_fine_tuned[k].append(ft_recall)
            print(row)

        print("-" * (38 + 48 * len(k_values)))

        # Print average scores
        avg_row = f"Average scores:  {' ' * 38} "
        total_orig_avg = 0
        total_ft_avg = 0
        for k in k_values:
            orig_avg = np.mean(avg_original[k])
            ft_avg = np.mean(avg_fine_tuned[k])
            total_orig_avg += orig_avg
            total_ft_avg += ft_avg
            avg_row += f" {orig_avg:.3f}{' ' * 16}{ft_avg:.3f}{' ' * 16}"
        print(avg_row)

        # Print total averages
        print(f"\nTotal average (Original):   {total_orig_avg / len(k_values):.3f}")
        print(f"Total average (Fine-tuned):   {total_ft_avg / len(k_values):.3f}")

        print("\nRatios (Original Recall : Fine-tuned Recall):")
        for query_type in metrics['original'].keys():
            ratio_row = f"{query_type:<40}"
            for k in k_values:
                orig_recall = np.mean(metrics['original'][query_type][k])
                ft_recall = np.mean(metrics['fine_tuned'][query_type][k])
                if ft_recall != 0:
                    ratio = orig_recall / ft_recall
                    ratio_row += f"{ratio:.3f}:1{' ' * 22}"
                else:
                    ratio_row += "inf:1" + " " * 22
            print(ratio_row)

        results = {
            "query_type": [],
            "k": [],
            "original_recall": [],
            "fine_tuned_recall": [],
            "ratio": []
        }

        for query_type in metrics['original'].keys():
            for k in k_values:
                orig_recall = np.mean(metrics['original'][query_type][k])
                ft_recall = np.mean(metrics['fine_tuned'][query_type][k])
                ratio = orig_recall / ft_recall if ft_recall != 0 else float('inf')

                results["query_type"].append(query_type)
                results["k"].append(k)
                results["original_recall"].append(orig_recall)
                results["fine_tuned_recall"].append(ft_recall)
                results["ratio"].append(ratio)

        # Add average scores
        for k in k_values:
            results["query_type"].append("Average")
            results["k"].append(k)
            results["original_recall"].append(np.mean(avg_original[k]))
            results["fine_tuned_recall"].append(np.mean(avg_fine_tuned[k]))
            results["ratio"].append(np.mean(avg_original[k]) / np.mean(avg_fine_tuned[k]))

        # Add total averages
        results["query_type"].append("Total Average")
        results["k"].append("All")
        results["original_recall"].append(total_orig_avg / len(k_values))
        results["fine_tuned_recall"].append(total_ft_avg / len(k_values))
        results["ratio"].append((total_orig_avg / len(k_values)) / (total_ft_avg / len(k_values)))

        return results

    def save_results(self, results, output_directory, num_works, num_test_works, k_values):
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Increment the result counter
        self.result_counter += 1

        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add test parameters and model config to results
        results["test_parameters"] = {
            "num_works": num_works,
            "num_test_works": num_test_works,
            "k_values": k_values,
            "timestamp": timestamp,
            "model_path": self.previous_model_path
        }
        results["model_config"] = self.model_config

        # Save as CSV
        csv_filename = f"evaluation_results_{timestamp}_{self.result_counter}.csv"
        csv_path = os.path.join(output_directory, csv_filename)
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ["query_type", "k", "original_recall", "fine_tuned_recall", "ratio", "timestamp", "model_path"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(results["query_type"])):
                writer.writerow({
                    "query_type": results["query_type"][i],
                    "k": results["k"][i],
                    "original_recall": results["original_recall"][i],
                    "fine_tuned_recall": results["fine_tuned_recall"][i],
                    "ratio": results["ratio"][i],
                    "timestamp": timestamp,
                    "model_path": self.previous_model_path
                })

        print(f"Results saved as CSV: {csv_path}")

        # Save as JSON
        json_filename = f"evaluation_results_{timestamp}_{self.result_counter}.json"
        json_path = os.path.join(output_directory, json_filename)
        with open(json_path, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)

        print(f"Results saved as JSON: {json_path}")

    def retrieve_and_save_works(self, num_works=10000):
        output_dir = r"D:\evaluate_search_vectordb"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "work_data.parquet")

        last_work_int_id = 1500000
        query = {"works_int_id": {"$gt": last_work_int_id}}
        works_cursor = self.works_collection.find(query).sort("works_int_id", 1).limit(num_works)

        work_data = []
        for work in tqdm(works_cursor, total=num_works, desc="Retrieving works"):
            work_info = self.extract_work_info(work)
            try:
                if work_info is not None:
                    primary_topic = work.get("primary_topic", {})
                    if primary_topic is None:
                        continue

                    field = primary_topic.get("field", {}).get("display_name", "")
                    subfield = primary_topic.get("subfield", {}).get("display_name", "")
                    domain = primary_topic.get("domain", {}).get("display_name", "")
                    topic = primary_topic.get("topic", {}).get("display_name", "")

                    [full_query], _ = self.create_queries(work_info, full_query_only=True)
                    original_embedding = self.original_model.encode(full_query)

                    work_data.append({
                        "work_id": work_info["work_id"],
                        "works_int_id": work_info["works_int_id"],
                        "domain": domain,
                        "field": field,
                        "subfield": subfield,
                        "topic": topic,
                        "work_string": full_query,
                        "original_embedding": original_embedding
                    })

            except Exception as e:
                print("Error: ", e)

        df = pd.DataFrame(work_data)
        df.to_parquet(output_file, index=False)
        print(f"Saved {len(df)} works to {output_file}")

    def print_evaluation_results_domain(self, metrics, k_values=[5, 10, 20, 50]):
        for domain in metrics['original'].keys():
            print(f"\nResults for domain: {domain}")
            header = f"{'Model':<15}"
            for k in k_values:
                header += f"{'Recall@' + str(k):<12}"
            print(header)
            print("-" * (15 + 12 * len(k_values)))

            for model_name in ['original', 'fine_tuned']:
                row = f"{model_name:<15}"
                for k in k_values:
                    recall = np.mean(metrics[model_name][domain]["recall"][k])
                    row += f"{recall:.3f}{' ' * 8}"
                print(row)

        print("\nOverall Results:")
        header = f"{'Model':<15}"
        for k in k_values:
            header += f"{'Avg Recall@' + str(k):<16}"
        print(header)
        print("-" * (15 + 16 * len(k_values)))

        for model_name in ['original', 'fine_tuned']:
            row = f"{model_name:<15}"
            for k in k_values:
                avg_recall = np.mean(
                    [np.mean(metrics[model_name][domain]["recall"][k]) for domain in metrics[model_name].keys()])
                row += f"{avg_recall:.3f}{' ' * 12}"
            print(row)

    @measure_time
    def run_test_domain(self, num_works=10000, num_test_works=500, k_values=[5, 10, 20, 50]):
        self.retrieve_and_save_works(num_works)
        metrics = self.evaluate_search_domain(num_test_works, k_values)
        self.print_evaluation_results_domain(metrics, k_values)

    @measure_time
    def run_test(self, num_works=12000, num_test_works=500, k_values=[5, 10, 20, 50]):
        work_infos = self.create_embeddings_and_index(num_works)
        metrics = self.evaluate_search(work_infos, num_test_works, k_values)
        results = self.print_evaluation_results(metrics, k_values)
        self.save_results(results, self.output_directory, num_works, num_test_works, k_values)


if __name__ == "__main__":
    k_values = [5, 10, 20, 50]

    # "D:\evaluate_search_vectordb\work_data.parquet"

    # 0.995, 0.990, 0.971 r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs32_2024_08_12\checkpoint-124985"

    # 0.993, 0.982, 0.964 r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_15\checkpoint-7936"

    # previous_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_15\checkpoint-256"
    # previous_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs32_2024_08_12\checkpoint-124985"
    # previous_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs32_2024_08_16\checkpoint-69632"
    # previous_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_15\checkpoint-10496"
    # r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_15\checkpoint-4352"
    # previous_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_15\checkpoint-7809"
    # previous_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs32_2024_08_12\checkpoint-124985"
    # previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\health_sciences_full\best_model_bs16_2024_08_30\checkpoint-34816"

    # previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\health_sciences_full\best_model_bs16_2024_08_31\checkpoint-163840"
    previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\social_sciences_full\best_model_bs16_2024_08_31\checkpoint-98304"
    #  checkpoint-99075   # 88064
    # previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\life_sciences_full\best_model_bs16_2024_08_31\checkpoint-61440"
    tester = VectorSearchAccuracyTest(previous_model_path, output_directory=r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\vectordb")

    # "C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\datasets_47598419"
    # {'eval_loss': 0.27789977192878723, 'eval_cosine_accuracy': 0.908, 'eval_dot_accuracy': 0.092, 'eval_manhattan_accuracy': 0.91, 'eval_euclidean_accuracy': 0.908, 'eval_max_accuracy': 0.91, 'eval_runtime': 2.307, 'eval_samples_per_second': 216.73, 'eval_steps_per_second': 13.871, 'epoch': 0.89}

    try:
        previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\social_sciences_full\best_model_bs16_2024_08_31\checkpoint-98304"

        tester.run_test(num_works=12000, num_test_works=1000, k_values=k_values)
        tester.run_test_domain(num_works=10000, num_test_works=5000, k_values=k_values)
        print("previous_model_path: ", previous_model_path)
    except Exception as e:
        print("Error: ", e)

    try:
        previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\physical_sciences_full\best_model_bs16_2024_08_30\checkpoint-182272"
        tester = VectorSearchAccuracyTest(previous_model_path,
                                          output_directory=r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\vectordb")
        tester.run_test(num_works=12000, num_test_works=1000, k_values=k_values)
        tester.run_test_domain(num_works=10000, num_test_works=5000, k_values=k_values)
        print("previous_model_path: ", previous_model_path)
    except Exception as e:
        print("Error: ", e)

    try:
        previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\health_sciences_full\best_model_bs16_2024_08_31\checkpoint-163840"
        tester = VectorSearchAccuracyTest(previous_model_path,
                                          output_directory=r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\vectordb")
        tester.run_test(num_works=12000, num_test_works=1000, k_values=k_values)
        tester.run_test_domain(num_works=10000, num_test_works=5000, k_values=k_values)
        print("previous_model_path: ", previous_model_path)
    except Exception as e:
        print("Error: ", e)

    try:
        previous_model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\life_sciences_full\best_model_bs16_2024_08_31\checkpoint-61440"
        tester = VectorSearchAccuracyTest(previous_model_path,
                                          output_directory=r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\vectordb")
        tester.run_test(num_works=12000, num_test_works=1000, k_values=k_values)
        tester.run_test_domain(num_works=10000, num_test_works=5000, k_values=k_values)
        print("previous_model_path: ", previous_model_path)
    except Exception as e:
        print("Error: ", e)

