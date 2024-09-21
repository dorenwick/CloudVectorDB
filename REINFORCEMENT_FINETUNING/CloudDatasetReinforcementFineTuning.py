import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pymongo import MongoClient

class CloudDatasetReinforcementFinetuning:
    """
    We want to search for works with particular title string length and author counts, and test for those. So, test on results with a single author, 2 authors, 3-10 authors, and 10+ authors.

    TODO: in addition to this class, we want a class that goes through every single work, and if they are not found in k20 nn search, we add the work_id to a list of all "not found" work_id's.
     In fact, i'd like you to create this class for us now. Here is how it will work. First, we are given an index, and we do the following queries on it:
     1 f"{title} {field}", 2 f"{trigram} {field} {subfield}", 3 f"{' '.join(author_names[:2])} {field}", 4 f"{author_names[0] if author_names else ''} {field} {subfield}", 5 full_query

     Now, we shall generate a table, or rather a parquet file, and for each one of these queries that doesn't result in the work_id being found in the top 20 nearest neighbours,
     we will record the work_id and the results as (0, 1) in each column (named after query type). We want columns for each query string of each query type as well,
      so we keep those query strings there for us. where 0 is not found, and 1 is found. We shall also put cited_by_count in another column, as we want that information to be made available.
       Then we will save the works as "missed_data.parquet".

    TODO: The next step shall be to generate a dataset of triplets (anchor, positive, negative) from the query_strings that didn't result in the work_id being found.
        To do this, we will create a vectordb of all of them, by encoding them, and then searching via each query_string and then making positive/negatives from them,
         (making sure there is reasonable distance between them all), until we have got every query_string built placed into triplets.
         This will require some effort and planning to figure out the specifics.

          TODO The next step shall be to fine-tune the encoder that was used to make the previous database, by incorporating the triplets into the last dataset (by adding them in),
            then shuffling the dataset, and training the model again. We may also want to remove data from the dataset where the knn got recall@1 100% score on every query_string. We could filter them out.
            This is something to figure out later on.

    TODO: The next step is to build the vectordb with new encodings and index it all over again. Consider This description.
        We have a class that





    """


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
        """
        We should only count a miss if the query actually has the data, so for example if it doesn't have any
        authors, then we will automatically count the query:             f"{' '.join(author_names[:2])} {field}",
        as a hit. (and make it 1 instead of 0).
        Similar logic should be used for titles.
        We ought to make a schema for specifying query types (title_query, author_query, both_query, topic_query, ect).


        :return:
        """

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
        ).limit(10_000)

        for work in tqdm(works_cursor, total=10_000, desc="Processing works"):
            work_id = work['id']
            works_int_id = work.get('works_int_id')
            if bad_results_count % 100 == 0:
                print("works_int_id: ", works_int_id)
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