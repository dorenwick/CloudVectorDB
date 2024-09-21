import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pymongo import MongoClient

class CloudDatasetReinforcementFinetuning:
    def __init__(self, index_path, model_path, mongodb_url, database_name):
        self.index_path = index_path
        self.model_path = model_path
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.index = None
        self.model = None
        self.mongo_client = None
        self.db = None
        self.works_collection = None

    def load_resources(self):
        print("Loading index...")
        self.index = faiss.read_index(self.index_path)

        print("Loading model...")
        self.model = SentenceTransformer(self.model_path)

        print("Connecting to MongoDB...")
        self.mongo_client = MongoClient(self.mongodb_url)
        self.db = self.mongo_client[self.database_name]
        self.works_collection = self.db['Works']

    def generate_queries(self, work):
        title = work.get('display_name', '')
        primary_topic = work.get("primary_topic", {})


        if primary_topic:
            field = primary_topic.get("field", {}).get("display_name", "")
            subfield = primary_topic.get("subfield", {}).get("display_name", "")
        else:
            field = ""
            subfield = ""
        author_names = [authorship.get('author', {}).get('display_name', '') for authorship in work.get('authorships', [])]


        if title:
            title_words = title.split()
        else:
            title_words = "science"

        trigram = " ".join(title_words[:min(3, len(title_words))])

        queries = [
            f"{title} {field}",
            f"{' '.join(author_names[:2])} {field}",
        ]

        return queries

    def search_similar_works(self, query_vector, k=20):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices[0], distances[0]

    def find_missed_works(self):
        missed_data = []
        good_results_count = 0
        bad_results_count = 0

        projection = {
            'id': 1,
            'works_int_id': 1,
            'display_name': 1,
            'primary_topic': 1,
            'authorships': 1,
            'cited_by_count': 1
        }

        works_cursor = self.works_collection.find(
            {'cited_by_count': {'$gte': 25}},
            projection
        ).limit(1000)

        for work in tqdm(works_cursor, total=1000, desc="Processing works"):
            work_id = work['id']
            works_int_id = work.get('works_int_id')
            queries = self.generate_queries(work)

            results = []
            query_strings = []

            for query in queries:
                query_vector = self.model.encode([query])[0]
                similar_indices, _ = self.search_similar_works(query_vector)
                retrieved_work_int_ids = [int(idx) + 1 for idx in similar_indices]
                found = works_int_id in retrieved_work_int_ids
                results.append(int(found))
                query_strings.append(query)

            if 0 in results:
                missed_data.append({
                    'work_id': work_id,
                    'works_int_id': works_int_id,
                    'query_1': query_strings[0],
                    'query_2': query_strings[1],
                    'result_1': results[0],
                    'result_2': results[1],
                    'cited_by_count': work['cited_by_count']
                })
                bad_results_count += 1
            else:
                good_results_count += 1

        missed_df = pl.DataFrame(missed_data)
        missed_df.write_parquet("missed_data.parquet")
        print("Saved missed works data to missed_data.parquet")

        print(f"Good results (found in all queries): {good_results_count}")
        print(f"Bad results (missed in at least one query): {bad_results_count}")
        print(f"Total works processed: {good_results_count + bad_results_count}")
        print(f"Percentage of good results: {good_results_count / (good_results_count + bad_results_count) * 100:.2f}%")

        return missed_df



    def run(self):
        self.load_resources()
        self.find_missed_works()

# Usage
if __name__ == "__main__":
    index_path = r"E:\HugeDatasetBackup\works_index_updated.bin"
    mongodb_url = 'mongodb://localhost:27017'
    database_name = 'OpenAlex'
    model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\best_model"

    finetuner = CloudDatasetReinforcementFinetuning(index_path, model_path, mongodb_url, database_name)
    finetuner.run()