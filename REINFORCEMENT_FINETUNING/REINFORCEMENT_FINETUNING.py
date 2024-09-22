

import os
import time
import faiss
import numpy as np
import random
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import polars as pl
from tqdm import tqdm


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper

class TestWorksIndex:
    """
    We want to search for works with particular title string length and author counts, and test for those.
    So, test on results with a single author, 2 authors, 3-10 authors, and 10+ authors.

    TODO: in addition to this class, we want a class that goes through every single work, and if they are not found in k20 nn search, we add the work_id
        to a list of all "not found" work_id's. In fact, i'd like you to create this class for us now.

    Here is how it will work. First, we are given an index, and we do the following queries on it:
           1 f"{title} {field}",
           2 f"{trigram} {field} {subfield}",
           3 f"{' '.join(author_names[:2])} {field}",
           4 f"{author_names[0] if author_names else ''} {field} {subfield}",
           5 full_query

    Now, we shall generate a table, or rather a parquet file, and for each one of these queries that doesn't result in the work_id being found
        in the top 20 nearest neighbours, we will record the work_id and the results as (0, 1) in each column (named after query type).
        We want columns for each query string of each query type as well, so we keep those query strings there for us.

        where 0 is not found, and 1 is found. We shall also put cited_by_count in another column, as we want that information to be made available.

    Then we will save the works as "missed_data.parquet".

    TODO: The next step shall be to generate a dataset of triplets (anchor, positive, negative) from the query_strings that didn't result in the work_id being found.
        To do this, we will create a vectordb of all of them, by encoding them, and then searching via each query_string and then making positve/negatives from them,
        (making sure there is reasonable distance between them all), until we have got every query_string built placed into triplets.
        This will require some effort and planning to figure oout the specifics.

    TODO The next step shall be to fine-tune the encoder that was used to make the previous database, by incorparating the triplets into the last dataset (by adding them in), then shuffling
        the dataset, and training the model again. We may also want to remove data from the dataset where the knn got recall@1 100% score on every query_string. We could filter them out.
        This is something to figure out later on.

    TODO: The next step is to build the vectordb with new encodings and index it all over again.




    """

    def __init__(self, index_path, mongodb_url, database_name, model_path):
        self.index_path = index_path
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.model_path = model_path
        self.index = None
        self.mongo_client = None
        self.db = None
        self.works_collection = None
        self.model = None


    def load_resources(self):
        print(f"Loading index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        print("Connecting to MongoDB")
        self.mongo_client = MongoClient(self.mongodb_url)
        self.db = self.mongo_client[self.database_name]
        self.works_collection = self.db['Works']

        print(f"Loading model from {self.model_path}")
        self.model = SentenceTransformer(self.model_path)

        self.works_collection = self.db['Works']
        self.life_sciences_collection = self.db['LifeSciences']
        self.physical_sciences_collection = self.db['PhysicalSciences']
        self.social_sciences_collection = self.db['SocialSciences']
        self.health_sciences_collection = self.db['HealthSciences']


    def extract_work_info(self, work_data):
        work_id = work_data.get("id")
        title = work_data.get("display_name", "")
        primary_topic = work_data.get("primary_topic", {})
        if primary_topic:
            field = primary_topic.get("field", {}).get("display_name", "")
            subfield = primary_topic.get("subfield", {}).get("display_name", "")
        else:
            field = ""
            subfield = ""

        author_names = []
        for authorship in work_data.get("authorships", [])[:20]:
            author = authorship.get("author", {})
            if "display_name" in author and "id" in author:
                author_names.append(author["display_name"])

        return {
            "work_id": work_id,
            "works_int_id": work_data.get("works_int_id"),
            "title": title,
            "field": field,
            "subfield": subfield,
            "author_names": author_names,
        }

    def create_queries(self, work_info, full_query_only=False):
        def clean_name(name):
            return ' '.join(name.strip().split()) if name else ""

        title = clean_name(work_info["title"])
        author_names = [clean_name(name) for name in work_info["author_names"] if name.strip()][:20]
        field = clean_name(work_info["field"])
        subfield = clean_name(work_info["subfield"])

        title_words = title.split()
        trigram = " ".join(title_words[:min(3, len(title_words))]) + " "

        full_query = f"{title} {' '.join(author_names)} {field} {subfield}"

        if full_query_only:
            return [full_query], full_query

        queries = [
            f"{title} {field}",
            f"{trigram} {field} {subfield}",
            f"{trigram} {field}",
            f"{' '.join(author_names[:2])} {field}",
            f"{author_names[0] if author_names else ''} {field} {subfield}",
            f"{author_names[0] if author_names else ''} {field}",
            full_query
        ]
        # if random.random() < 0.01:
        #     print("queries", queries)

        queries = [q.strip() for q in queries if q.strip()]
        while len(queries) < 7:
            queries.append(" ")

        return queries[:7], full_query

    def encode_string(self, string):
        return self.model.encode([string])[0]

    # @measure_time
    def search_similar_works(self, query_vector, k=2500):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices[0], distances[0]

    def calculate_recall_at_k(self, relevant_items, retrieved_items, k):
        return len(set(relevant_items) & set(retrieved_items[:k])) / len(relevant_items)

    @measure_time
    def evaluate_search(self, work_infos, num_test_works=100, k_values=[5, 10, 20, 50, 250, 2500]):
        test_works = random.sample(work_infos, min(num_test_works, len(work_infos)))
        query_types = [
            "Full title + field",
            "Trigram from title + field + subfield",
            "Trigram from title + field",
            "First 2 authors + field",
            "One author + field + subfield",
            "Author + field",
            "Full query"
        ]

        metrics = defaultdict(lambda: defaultdict(list))
        max_k = max(k_values)

        for work in tqdm(test_works, desc="Evaluating search performance"):
            queries, _ = self.create_queries(work)
            work_int_id = work['works_int_id']

            for i, query in enumerate(queries):
                query_vector = self.encode_string(query)
                similar_indices, _ = self.search_similar_works(query_vector, k=max_k)
                retrieved_work_int_ids = [int(idx) + 1 for idx in similar_indices]

                for k in k_values:
                    recall = self.calculate_recall_at_k([work_int_id], retrieved_work_int_ids, k)
                    metrics[query_types[i]][k].append(recall)

        return metrics

    def print_evaluation_results(self, metrics, k_values=[5, 10, 20, 50, 250, 2500]):
        print("\nEvaluation Results:")
        print("-" * 80)
        print(f"{'Query Type':<40} " + " ".join([f"Recall@{k:<8}" for k in k_values]))
        print("-" * 80)

        for query_type, recalls in metrics.items():
            row = f"{query_type:<40} "
            for k in k_values:
                avg_recall = np.mean(recalls[k])
                row += f"{avg_recall:.4f}    "
            print(row)

        print("-" * 80)
        print("Average Recall Across All Query Types:")
        for k in k_values:
            avg_recall = np.mean([np.mean(recalls[k]) for recalls in metrics.values()])
            print(f"Recall@{k}: {avg_recall:.4f}")



    def get_works_with_condition(self, condition, num_test_works):
        collection_stats = self.works_collection.aggregate([{"$collStats": {"count": {}}}]).next()
        total_works = collection_stats.get("count", 0)
        start_id = random.randint(100_000, max(1, (total_works - num_test_works)) // 2)

        query = {
            "$and": [
                {"works_int_id": {"$gte": start_id}},
                condition
            ]
        }

        works = list(self.works_collection.find(query).limit(num_test_works))

        # If we don't have enough works, get the remaining from the beginning
        if len(works) < num_test_works:
            remaining = num_test_works - len(works)
            additional_works = list(self.works_collection.find(
                {"$and": [{"works_int_id": {"$lt": start_id}}, condition]}
            ).limit(remaining))
            works.extend(additional_works)

        return works

    @measure_time
    def get_works_with_abstracts(self, num_test_works):
        print("hi 1")
        condition = {"abstract_inverted_index": {"$exists": True, "$ne": {}}}
        return self.get_works_with_condition(condition, num_test_works)

    @measure_time
    def get_works_with_topics(self, num_test_works):
        print("hi 2")
        condition = {"primary_topic": {"$exists": True, "$ne": None}}
        return self.get_works_with_condition(condition, num_test_works)

    @measure_time
    def get_works_with_abstracts_and_topics(self, num_test_works):
        print("hi 3")
        condition = {
            "abstract_inverted_index": {"$exists": True, "$ne": {}},
            "primary_topic": {"$exists": True, "$ne": None}
        }
        return self.get_works_with_condition(condition, num_test_works)

    @measure_time
    def get_works_by_domain(self, domain, num_test_works):
        condition = {"primary_topic.domain.display_name": domain}
        return self.get_works_with_condition(condition, num_test_works)

    @measure_time
    def get_works_life_sciences(self, num_test_works):
        return self.get_works_by_domain("Life Sciences", num_test_works)

    @measure_time
    def get_works_physical_sciences(self, num_test_works):
        return self.get_works_by_domain("Physical Sciences", num_test_works)

    @measure_time
    def get_works_social_sciences(self, num_test_works):
        return self.get_works_by_domain("Social Sciences", num_test_works)

    @measure_time
    def get_works_health_sciences(self, num_test_works):
        return self.get_works_by_domain("Health Sciences", num_test_works)

    @measure_time
    def get_works_by_field(self, field, num_test_works):
        condition = {"primary_topic.field.display_name": field}
        return self.get_works_with_condition(condition, num_test_works)

    @measure_time
    def get_works_with_citation_count(self, num_test_works, condition):
        return self.get_works_with_condition(condition, num_test_works)

    @measure_time
    def get_works_with_citation_count_1(self, num_test_works):
        condition = {"cited_by_count": 1}
        return self.get_works_with_citation_count(num_test_works, condition)

    @measure_time
    def get_works_with_citation_count_over_5(self, num_test_works):
        condition = {"cited_by_count": {"$gt": 5}}
        return self.get_works_with_citation_count(num_test_works, condition)

    @measure_time
    def get_works_with_citation_count_over_25(self, num_test_works):
        condition = {"cited_by_count": {"$gt": 25}}
        return self.get_works_with_citation_count(num_test_works, condition)

    @measure_time
    def get_works_with_citation_count_over_100(self, num_test_works):
        condition = {"cited_by_count": {"$gt": 100}}
        return self.get_works_with_citation_count(num_test_works, condition)

    @measure_time
    def get_works_by_author_count(self, num_test_works, condition):
        return self.get_works_with_condition(condition, num_test_works)

    @measure_time
    def get_works_with_one_author(self, num_test_works):
        condition = {"authorships": {"$size": 1}}
        return self.get_works_by_author_count(num_test_works, condition)

    @measure_time
    def get_works_with_two_authors(self, num_test_works):
        condition = {"authorships": {"$size": 2}}
        return self.get_works_by_author_count(num_test_works, condition)

    @measure_time
    def get_works_with_three_to_ten_authors(self, num_test_works):
        condition = {"$and": [
            {"authorships": {"$exists": True}},
            {"authorships.2": {"$exists": True}},  # At least 3 authors
            {"authorships.10": {"$exists": False}}  # Less than 11 authors
        ]}
        return self.get_works_by_author_count(num_test_works, condition)

    @measure_time
    def get_works_with_eleven_plus_authors(self, num_test_works):
        condition = {"authorships.10": {"$exists": True}}  # At least 11 authors
        return self.get_works_by_author_count(num_test_works, condition)

    @measure_time
    def get_random_works(self, num_test_works):
        collection_stats = self.works_collection.aggregate([{"$collStats": {"count": {}}}]).next()
        total_works = collection_stats.get("count", 0)
        start_id = random.randint(1, max(1, total_works - num_test_works))
        return list(self.works_collection.find({"works_int_id": {"$gte": start_id}}).limit(num_test_works))


    @measure_time
    def run_test(self, works, test_name, num_test_works, k_values):
        work_infos = [self.extract_work_info(work) for work in works]
        metrics = self.evaluate_search(work_infos, num_test_works=num_test_works, k_values=k_values)
        print(f"\nEvaluation Results for {test_name}:")
        self.print_evaluation_results(metrics, k_values)

    @measure_time
    def run(self, num_test_works=100):
        self.load_resources()

        collection_stats = self.works_collection.aggregate([{"$collStats": {"count": {}}}]).next()
        total_works = collection_stats.get("count", 0)
        print(f"Total works: {total_works}")

        k_values = [5, 10, 20, 50, 250, 2500]

        works_with_topics = self.get_works_with_topics(num_test_works)
        self.run_test(works_with_topics, "Works with Topics", num_test_works, k_values)

        # Test works with abstracts
        works_with_abstracts = self.get_works_with_abstracts(num_test_works)
        self.run_test(works_with_abstracts, "Works with Abstracts", num_test_works, k_values)


        # Test works with both abstracts and topics
        works_with_both = self.get_works_with_abstracts_and_topics(num_test_works)
        self.run_test(works_with_both, "Works with Abstracts and Topics", num_test_works, k_values)


        # Test works by domain
        domain_methods = [
            (self.get_works_life_sciences, "Life Sciences"),
            (self.get_works_physical_sciences, "Physical Sciences"),
            (self.get_works_social_sciences, "Social Sciences"),
            (self.get_works_health_sciences, "Health Sciences"),
        ]


        for get_works_method, domain_name in domain_methods:
            works_by_domain = get_works_method(num_test_works)
            self.run_test(works_by_domain, f"Works in {domain_name} Domain", num_test_works, k_values)

        # Test works by field
        fields = [
            "Health Professions", "Agricultural and Biological Sciences", "Neuroscience", "Energy",
            "Materials Science", "Nursing", "Business, Management and Accounting", "Engineering",
            "Environmental Science", "Economics, Econometrics and Finance",
            "Biochemistry, Genetics and Molecular Biology", "Medicine", "Chemistry",
            "Arts and Humanities", "Earth and Planetary Sciences", "Psychology",
            "Pharmacology, Toxicology and Pharmaceutics", "Computer Science", "Dentistry",
            "Mathematics", "Veterinary", "Chemical Engineering", "Social Sciences",
            "Decision Sciences", "Immunology and Microbiology", "Physics and Astronomy"
        ]

        for field in fields:
            works_by_field = self.get_works_by_field(field, num_test_works)
            self.run_test(works_by_field, f"Works in {field} Field", num_test_works, k_values)

        # New tests for citation counts
        citation_count_methods = [
            (self.get_works_with_citation_count_1, "Works with Citation Count 1"),
            (self.get_works_with_citation_count_over_5, "Works with Citation Count Over 5"),
            (self.get_works_with_citation_count_over_25, "Works with Citation Count Over 25"),
            #(self.get_works_with_citation_count_over_100, "Works with Citation Count Over 100"),
        ]

        for get_works_method, test_name in citation_count_methods:
            works = get_works_method(num_test_works)
            self.run_test(works, test_name, num_test_works, k_values)


        # New tests for author counts
        author_count_methods = [
            (self.get_works_with_one_author, "Works with 1 Author"),
            (self.get_works_with_two_authors, "Works with 2 Authors"),
            (self.get_works_with_three_to_ten_authors, "Works with 3-10 Authors"),
            (self.get_works_with_eleven_plus_authors, "Works with 11+ Authors"),
        ]

        for get_works_method, test_name in author_count_methods:
            works = get_works_method(num_test_works)
            if works:
                self.run_test(works, test_name, len(works), k_values)
            else:
                print(f"No works found for {test_name}")

        random_works = self.get_random_works(num_test_works)
        self.run_test(random_works, "Random Works", num_test_works, k_values)

    def create_work_index_test_data(self, num_works=100_000):
        print(f"Creating work index test data with {num_works} works...")

        print("Connecting to MongoDB")
        self.mongo_client = MongoClient(self.mongodb_url)
        self.db = self.mongo_client[self.database_name]
        self.works_collection = self.db['Works']

        collection_stats = self.works_collection.aggregate([{"$collStats": {"count": {}}}]).next()
        total_works = collection_stats.get("count", 0)
        random_works = self.works_collection.aggregate([
            {"$sample": {"size": num_works}}
        ])

        data = []
        for work in tqdm(random_works, total=num_works, desc="Processing works"):
            work_info = self.extract_work_info(work)
            queries, full_query = self.create_queries(work_info)

            primary_topic = work.get("primary_topic") or {}
            domain = primary_topic.get("domain") or {}
            field = primary_topic.get("field") or {}
            subfield = primary_topic.get("subfield") or {}


            title = work_info.get("title", "")  # Use work_info instead of work

            title_length = len(title) if title else 0

            row = {
                'work_id': work.get('id', ''),
                'works_int_id': work.get('works_int_id', 0),
                'query_full_title_field': queries[0],
                'query_trigram_field_subfield': queries[1],
                'query_trigram_field': queries[2],
                'query_first_two_authors_field': queries[3],
                'query_one_author_field_subfield': queries[4],
                'query_one_author_field': queries[5],
                'query_full': queries[6],
                'query_all': full_query,
                'title': title,
                'title_length': title_length,
                'num_authors': len(work.get('authorships', [])),
                'domain': domain.get('display_name', ''),
                'field': field.get('display_name', ''),
                'subfield': subfield.get('display_name', ''),
                'has_abstract': bool(work.get('abstract_inverted_index')),
                'citation_count': work.get('cited_by_count', 0),
            }
            data.append(row)

        # Create Polars DataFrame
        df = pl.DataFrame(data)

        # Add columns for specific conditions
        df = df.with_columns([
            (pl.col('num_authors') == 1).alias('has_one_author'),
            (pl.col('num_authors') == 2).alias('has_two_authors'),
            ((pl.col('num_authors') >= 3) & (pl.col('num_authors') <= 10)).alias('has_three_to_ten_authors'),
            (pl.col('num_authors') > 10).alias('has_eleven_plus_authors'),
            (pl.col('citation_count') == 1).alias('has_one_citation'),
            (pl.col('citation_count') > 5).alias('has_over_five_citations'),
            (pl.col('citation_count') > 25).alias('has_over_twentyfive_citations'),
            (pl.col('citation_count') > 100).alias('has_over_hundred_citations'),
        ])

        # Add columns for each field
        fields = [
            "Health Professions", "Agricultural and Biological Sciences", "Neuroscience", "Energy",
            "Materials Science", "Nursing", "Business, Management and Accounting", "Engineering",
            "Environmental Science", "Economics, Econometrics and Finance",
            "Biochemistry, Genetics and Molecular Biology", "Medicine", "Chemistry",
            "Arts and Humanities", "Earth and Planetary Sciences", "Psychology",
            "Pharmacology, Toxicology and Pharmaceutics", "Computer Science", "Dentistry",
            "Mathematics", "Veterinary", "Chemical Engineering", "Social Sciences",
            "Decision Sciences", "Immunology and Microbiology", "Physics and Astronomy"
        ]

        for field in fields:
            df = df.with_columns((pl.col('field') == field).alias(f'is_{field.lower().replace(" ", "_")}'))

        print(df.head(100))
        print(df.tail(100))
        output_file = "work_index_test_data.parquet"

        output_directory = os.path.join(r"E:\HugeDatasetBackup\test_data", output_file)
        # Save to parquet file

        df.write_parquet(output_directory)
        print(f"Saved work index test data to {output_directory}")

        return df

if __name__ == "__main__":
    index_path = r"E:\HugeDatasetBackup\works_index_updated.bin"
    mongodb_url = 'mongodb://localhost:27017'
    database_name = 'OpenAlex'
    model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\best_model"

    tester = TestWorksIndex(index_path, mongodb_url, database_name, model_path)
    tester.create_work_index_test_data()
    # tester.run(num_test_works=100)
    # tester.run(num_test_works=500)