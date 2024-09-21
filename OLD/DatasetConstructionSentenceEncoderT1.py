import gc
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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from transformers import AutoTokenizer

latex = "Your LaTeX code here"
text = LatexNodes2Text().latex_to_text(latex)

# from SearchTest.VectorSearchAccuracyTest import VectorSearchAccuracyTest

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


    TODO: Create a separate method that goes over all the works with title string but not topic string, and
        creates two augmentations of them, as well as encodes them and builds pairs of them.
        So, we will want to make a method that searches over mongodb using projection to filter for works that
        have no topic, but have a title string or an authors string.
        We can create a small vectordb and then add them as pairs to our dataset.

    TODO: Given our no topic_works all parquet file we made here. We would actually like to add it to the vectordb we construct, for






    TODO: We need to ensure every single author_id appears, at least once. Try and get at least two counts.
    TODO: The goal will be to train a title, authors, field, subfield, topic, keywords string, and then finetune for:

    1: [title + field + subfield] + authors + topic + keywords
    2: [authors + field + subfield] + title + topic + keywords
    3: [field + subfield + topic + keywords] + topic + authors

    We shall also make 22million parameter models for:
        1: title + field + subfield
        2: authors + field + subfield


    TODO: =================================================================




    TODO: Refactor the index for gpu-processing.
        We need to learn how to load up the vector index to be trained on multiple gpu's.

    TODO: We wish to mix in work objects that have titles but do not have primary topics.

    TODO: Fine-tuning,




    TODO: We need to setup a system for generating datasets for fine-tuning.
        THIS will be an easy thing to setup. We will need to filter by source. We can just do that and be fine.


    TODO: We have to build the meta-data vectors. Make sure that that the mapping is consistent with openalex integer-id2label and label2id encodings.

    # Final schema:
    # Schema([('work_id_one', String), ('full_string_one', String), ('work_id_two', String), ('full_string_two', String), ('common_uni_grams', List(String)), ('common_bi_grams', List(String)), ('common_field', Boolean), ('common_subfield', Boolean), ('total_score', Float64), ('label', String), ('label_int', Int64), ('p_value', Float64), ('unigram_score', Float64), ('bigram_score', Float64), ('sum_gram_score', Float64), ('field_score', Float64), ('subfield_score', Float64), ('source', String)])
    # First 20 rows of final dataframe:



    TODO: We need to seriously fix this whole thing up for paths, and directories.
         We want to throw paths in here that will either be cloud based paths, or paths to run locally.

    TODO: We could make this whole thing speed up by using CAGRA, since its going to be run on linux.

    TODO: This system will run on cuda 12.2

    TODO: We need the compute distance function to be used for the author pairs as well.

    TODO: Try Implementing Polars in places where it shall help. We will have to test polars here locally.
         Try implementing numpy instead of pandas when we know the datatypes.

    TODO: We may wish to filter out first names or initials from Author names in this class, as well
        as any words that basically aren't high enough scores.


    We will be adjusting this class so it constructs three kinds of triplet datasets.
    One shall be of the variant where we make this string:

    TODO: full_string = f"{row['title_string']} {row['authors_string']} {row['field_string']} {row['subfield_string']}"

    Another shall be of the variant where we make this string:

    TODO: full_string = f"{row['title_string']} {row['authors_string']} {row['field_string']} {row['subfield_string']} {row['topic_string']} {row['keyphrases_string']}"

    Another shall be of the variant where we make this string:

    TODO: full_string = f"{row['field_string']} {row['subfield_string']} {row['topic_string']} {row['keyphrases_string']}"

    TODO: full_string = f" {row['authors_string']} {row['field_string']} {row['subfield_string']} {row['topic_string']} {row['keyphrases_string']}"

    TODO: full_string = f"{row['title_string']} {row['field_string']} {row['subfield_string']} {row['topic_string']} {row['keyphrases_string']}"


    TODO: We will make a refined triplet dataset, and a larger one.
        The refined one shall have no work_id duplicates in the triplet.

    TODO: We shall create triplet datasets for fine-tuning to particular

    TODO: We shall want to do some more sophisticated author augmentations.
        Like switch display_name with alternative names is one.
        We also may want to just put last names in, and get rid of the initials.
        That would work well.

    TODO: Instead of randomly generated ngrams, we could use lookups for key phrases.

    TODO: Currently we have knn_searched strings, common_tile_names, and common_author_names.

    TODO: Some extra issues we will encounter is this: latex in the title, and html as well. This may obscure results.
        Therefore, we will want to augment some strings to have their latex removed.

    TODO: We will be using pre-processed key phrases and short_unigrams, short_bigrams.

    TODO: We will want to modify the scoring system a lot probably.

    TODO: We will need to decide on curriculum learning at a later point.


    TODO: We want a much better system for selecting pairs and triplets for our refined dataset.
        One way is to encode and compare similarities for anchor, positive, negative and try and ensure 0.5% similarity distance
        between anchor, positive, and 1% similarity distance between anchor, negative. and 0.5% similarity distance between positive, negative.
        To do this, we retrieve the similarities when we do knn vector retrieval,

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
        self.id_mapping_works_file = os.path.join(self.datasets_directory, "id_mapping_works.parquet")
        self.index_works_file = os.path.join(self.datasets_directory, "index_works.bin")
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

        # Model initialization
        self.model = SentenceTransformer(self.model_path)
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
        if self.run_params.get('load_and_print_data', False):
            self.load_and_print_data()

        if self.run_params.get('collect_all_works_metadata', False):
            self.collect_all_works_metadata(abstract_include=False)

        if self.run_params.get('restructure_common_authors', False):
            self.restructure_common_authors()

        if self.run_params.get('restructure_augmented_data', False):
            self.restructure_augmented_data()

        if self.run_params.get('create_sentence_embeddings', False):
            self.create_sentence_embeddings(works_batch_size=100_000)

        if self.run_params.get('build_vector_index', False):
            self.build_vector_index()

        if self.run_params.get('generate_training_pairs', False):
            self.generate_training_pairs(batch_size=4096, initial_k=128, distance_threshold=0.1)

        if self.run_params.get('create_common_title_works', False):
            self.create_common_title_works()

        if self.run_params.get('generate_all_work_id_pairs_dataset', False):
            self.generate_all_work_id_pairs_dataset(sort_by_distance=True)

    @measure_time
    def load_and_print_data(self):
        parquet_files = [
            self.works_augmented_data_file,
            self.works_common_authors_file,
            self.works_all_collected_file,
            self.works_knn_search_file,
            self.triplet_work_ids_only_file
        ]

        for file in parquet_files:
            if os.path.exists(file):
                df = pl.read_parquet(file)
                print(f"\nFile: {os.path.basename(file)}")
                print("Schema:")
                print(df.dtypes.to_string())
                print("\nHead (50 rows):")
                print(df.head(50).to_string())
                print("\nTail (50 rows):")
                print(df.tail(50).to_string())
                del df
                gc.collect()
            else:
                print(f"File not found: {file}")

    def print_memory_usage(self, location):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage at {location}: {memory_info.rss / 1024 / 1024:.2f} MB")

    def create_works_notopic_all(self):
        print("Creating works_notopic_all.parquet file...")

        self.establish_mongodb_connection()

        # Define the projection to fetch only the required fields
        projection = {
            "works_int_id": 1,
            "id": 1,
            "display_name": 1,
            "authorships": 1,
            "primary_topic": 1,
            "cited_by_count": 1,
            "_id": 0
        }

        # Define the query to filter works without a primary topic
        query = {
            "$or": [
                {"primary_topic": {"$exists": False}},
                {"primary_topic": None}
            ],
            "$and": [
                {"$or": [
                    {"display_name": {"$exists": True, "$ne": ""}},
                    {"authorships": {"$exists": True, "$ne": []}}
                ]}
            ]
        }

        cursor = self.mongodb_works_collection.find(query, projection).batch_size(10000)

        works_data = []
        for work in tqdm(cursor, desc="Processing works without topic"):
            work_id = work.get('id')
            title = work.get('display_name', '')
            cited_by_count = work.get('cited_by_count', 0)

            author_names = []
            for authorship in work.get('authorships', []):
                author = authorship.get('author', {})
                if 'display_name' in author:
                    author_names.append(author['display_name'])

            authors_string = ' '.join(author_names)

            if title or authors_string:
                works_data.append({
                    'work_id': work_id,
                    'work_int_id': work.get('works_int_id'),
                    'title_string': title,
                    'authors_string': authors_string,
                    'cited_by_count': cited_by_count
                })

            if len(works_data) % 100000 == 0:
                print(f"Processed {len(works_data)} works")
                self.print_memory_usage(f"batch {len(works_data)}")

        self.close_mongodb_connection()

        # Create a Polars DataFrame
        df = pl.DataFrame(works_data)

        # Save the DataFrame as a parquet file
        output_file = os.path.join(self.datasets_directory, "works_notopic_all.parquet")
        df.write_parquet(output_file)

        print(f"Created works_notopic_all.parquet with {len(df)} works")
        return output_file


    @measure_time
    def collect_all_works_metadata(self, abstract_include=False):
        """


        TODO: please fix the save common authors file. We arnt doing it anymore.

        TODO:
            "contains_title"
            "contains_topic"
            "contains_authors"
            "char_len_above_20"
            "char_len_above_50"
            "field_type" (0 to 25)
            "subfield_type" (0 to ?)
            "topic_type" (0 to ?)

        TODO: We may have to stop at 100M works to avoid memory overload.

        TODO: We want to save some extra dataframes when we do this, because we have the opportunity.
            So for example, we could make a parquet file for just works and work_int_id and save that.
            This will be reusable alter on. Save that as a parquet file (initially using polars).

        :param abstract_include:
        :return:
        """


        self.establish_mongodb_connection()
        self.print_memory_usage("after establishing MongoDB connection")

        print("Collecting metadata for all works...")

        total_processed = 0

        batch_size = 100_000
        batch_count = 0
        new_rows = []
        batch_files = []

        # Define projection to fetch only the required fields
        projection = {
            "works_int_id": 1,
            "id": 1,
            "display_name": 1,
            "primary_topic": 1,
            "cited_by_count": 1,
            "authorships": 1,
            "_id": 0  # Exclude the default _id field
        }

        if abstract_include:
            projection["abstract_inverted_index"] = 1

        # Use batch_size and projection in the find method
        cursor = self.mongodb_works_collection.find(
            projection=projection
        ).sort("works_int_id", 1).batch_size(batch_size)

        for work in tqdm(cursor, desc="Processing works"):
            work_int_id = work.get('works_int_id')
            work_id = work.get('id')
            title = work.get('display_name')
            primary_topic = work.get('primary_topic', {})
            cited_by_count = work.get('cited_by_count', 0)

            if not title or not primary_topic:
                continue

            field = primary_topic.get('field', {}).get('display_name')
            subfield = primary_topic.get('subfield', {}).get('display_name')

            if not field or not subfield:
                continue

            author_names = []
            author_ids = []
            for authorship in work.get('authorships', []):
                author = authorship.get('author', {})
                if 'display_name' in author and 'id' in author:
                    author_names.append(author['display_name'])
                    author_ids.append(author['id'])


            authors_string = ' '.join(author_names)
            text_for_grams = f"{title} {authors_string}"

            if len(text_for_grams) < 5:
                continue

            unigrams = text_for_grams.lower().split()
            bigrams = [f"{unigrams[i]} {unigrams[i + 1]}" for i in range(len(unigrams) - 1)]

            if abstract_include:
                abstract_string = self.reconstruct_abstract(work.get('abstract_inverted_index', {}))
            else:
                abstract_string = ''

            new_rows.append({
                'work_id': work_id,
                'work_int_id': work_int_id,
                'title_string': title,
                'authors_string': authors_string,
                'author_names': author_names,
                'field_string': field,
                'subfield_string': subfield,
                'abstract_string': abstract_string,
                'unigrams': unigrams,
                'bigrams': bigrams,
                'cited_by_count': cited_by_count
            })

            total_processed += 1

            if len(new_rows) >= batch_size:
                batch_file = self.save_batch_to_parquet(new_rows, batch_count)
                batch_files.append(batch_file)
                new_rows = []
                batch_count += 1

            if total_processed % 100_000 == 0:
                self.print_memory_usage(f"for {total_processed}")
                print(f"Processed {total_processed} works")
                print(f"Total {self.num_works_collected}")
                gc.collect()

            if total_processed >= self.num_works_collected:
                break

        # Save any remaining rows
        if new_rows:
            batch_file = self.save_batch_to_parquet(new_rows, batch_count)
            batch_files.append(batch_file)

        self.close_mongodb_connection()

        # Save the final concatenated DataFrame
        output_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")

        self.print_memory_usage("Before concatenation")

        # Concatenate all batch files
        final_df = self.concatenate_parquet_files(batch_files)

        self.print_memory_usage("After concatenation")

        final_df.write_parquet(output_file)

        print(f"Saved final concatenated Polars DataFrame to {output_file}")
        self.print_memory_usage("After saving final concatenated Polars DataFrame")

    def save_batch_to_parquet(self, rows, batch_number):
        schema = {
            'work_id': pl.Utf8,
            'work_int_id': pl.Int32,
            'title_string': pl.Utf8,
            'authors_string': pl.Utf8,
            'author_names': pl.List(pl.Utf8),
            'field_string': pl.Utf8,
            'subfield_string': pl.Utf8,
            'abstract_string': pl.Utf8,
            'unigrams': pl.List(pl.Utf8),
            'bigrams': pl.List(pl.Utf8),
            'cited_by_count': pl.Int32
        }

        df = pl.DataFrame(rows, schema=None)
        batch_file = os.path.join(self.datasets_directory, f"works_batch_{batch_number}.parquet")
        df.write_parquet(batch_file)
        print(f"Saved batch {batch_number} to {batch_file}")
        return batch_file

    def concatenate_parquet_files(self, file_list):
        dfs = [pl.read_parquet(file) for file in file_list]
        concatenated_df = pl.concat(dfs)
        return concatenated_df


    @measure_time
    def restructure_common_authors(self):
        """
        TODO: We wish to start by removing all of the duplicates. We could do this before we run this method actually.


        :return:
        """

        common_authors_file = os.path.join(self.datasets_directory, "works_common_authors.parquet")

        print("Reading common authors file...")
        df = pl.read_parquet(common_authors_file)

        # TODO: This line is for testing.
        df = df[:100_000]

        print("Original shape:", df.shape)

        print("Removing duplicate work_id pairs...")
        df_filtered = df.filter(pl.col("work_id_one") != pl.col("work_id_two"))

        print("Filtered shape:", df_filtered.shape)

        print("Saving filtered data...")
        filtered_file = os.path.join(self.datasets_directory, "works_common_authors_filtered.parquet")
        df_filtered.write_parquet(filtered_file)

        print(f"Saved {df_filtered.shape[0]} common author pairs to {filtered_file}")

        common_authors_file = os.path.join(self.datasets_directory, "works_common_authors.parquet")
        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")

        print("Filtering common authors file...")

        # Read the parquet files
        df = pl.read_parquet(common_authors_file)
        works_df = pl.read_parquet(works_file)

        initial_rows = df.shape[0]
        print(f"Initial number of rows: {initial_rows}")

        # Create a set to keep track of encountered work_ids
        encountered_work_ids = set()

        # Function to check if a row should be kept
        def keep_row(row):
            work_id_one, work_id_two = row['work_id_one'], row['work_id_two']
            if work_id_one not in encountered_work_ids and work_id_two not in encountered_work_ids:
                encountered_work_ids.add(work_id_one)
                encountered_work_ids.add(work_id_two)
                return True
            return False

        # Apply the filtering
        filtered_df = df.filter(pl.struct(['work_id_one', 'work_id_two']).map_elements(lambda x: keep_row(x)))

        # Fetch work details
        all_work_ids = set(filtered_df['work_id_one'].to_list() + filtered_df['work_id_two'].to_list())
        work_details = self.fetch_work_details(all_work_ids, works_df, truncated=False, filter_works=True)

        # Process common elements
        pairs = list(zip(filtered_df['work_id_one'].to_list(), filtered_df['work_id_two'].to_list()))
        common_unigrams, common_bigrams, common_fields, common_subfields = self.process_common_elements(work_details,
                                                                                                        pairs)

        gc.collect()

        unigrams_df, bigrams_df = self.load_ngrams()

        # Vectorized processing of common elements
        vectorized_unigrams = self.vectorized_common_unigrams(common_unigrams)
        vectorized_bigrams = self.vectorized_common_bigrams(common_bigrams)
        vectorized_fields = self.vectorized_common_fields(common_fields)
        vectorized_subfields = self.vectorized_common_subfields(common_subfields)

        # Prepare data for insertion
        insert_data = []
        for i, (work1_id, work2_id) in enumerate(pairs):
            work1 = work_details.get(work1_id, {})
            work2 = work_details.get(work2_id, {})
            if work1 and work2:
                # Check if title strings are different before inserting
                if work1.get('title_string', '') != work2.get('title_string', ''):
                    insert_data.append({
                        'work_id_one': work1_id,
                        'full_string_one': f"{work1.get('title_string', '')} {work1.get('authors_string', '')} {work1.get('field_string', '')} {work1.get('subfield_string', '')}",
                        'work_id_two': work2_id,
                        'full_string_two': f"{work2.get('title_string', '')} {work2.get('authors_string', '')} {work2.get('field_string', '')} {work2.get('subfield_string', '')}",
                        'common_uni_grams': vectorized_unigrams[i],
                        'common_bi_grams': vectorized_bigrams[i],
                        'common_field': bool(vectorized_fields[i]),
                        'common_subfield': bool(vectorized_subfields[i]),
                        'total_score': 0.0,
                        'label': '',
                        'label_int': 0,
                        'p_value': 0.0
                    })

        # Calculate total scores
        insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)

        # Convert insert_data back to DataFrame
        filtered_df = pl.DataFrame(insert_data)

        filtered_df = filtered_df.with_columns(pl.lit('works_common_authors').alias('source'))

        print("\nFinal schema:")
        print(filtered_df.schema)
        print("\nFirst 20 rows of final dataframe:")
        print(filtered_df.head(20))

        final_rows = filtered_df.shape[0]
        print(f"Final number of rows: {final_rows}")
        print(f"Removed {initial_rows - final_rows} rows")

        common_authors_file_filtered = os.path.join(self.datasets_directory, "works_common_authors_filtered.parquet")

        # Save the filtered DataFrame
        filtered_df.write_parquet(common_authors_file_filtered)
        print(f"Filtered common authors file saved to {common_authors_file_filtered}")

    @measure_time
    def create_augmented_data(self):
        """

        TODO: We want to implement some more code here.
              We want the following. We want to make the augmentation method not probabilistic, but determined based
              off a priority system.

        TODO: For works that have a title_string >= 5 unigrams when tokenized via split(' '), we will make a random
            ngram like this:
                n = random.randint(2, len(title_words))
                m = random.randint(1, n)
                trigram = ' '.join(title_words[n - m:n])
                plus we will also find any keyphrases in the title, looking in the keywords dictionary that we shall have to make.
                If we find any, we will replace title with the keyphrases joined together as a string.
                + field (0.5%)
                + subfield (0.25%)

        TODO: For works that have between 1 and 2 authors, we will make one of the following:
            field (0.05 %)
            subfield (0.05 %)
            author + field (0.45 %)
            author + subfield (0.45 %)

        TODO: For works that have between 3 and 6 authors, we will make one of the following:
            author + field (0.15%)
            author + subfield (0.15%)
            two authors + field (0.2%)
            two authors + subfield (0.15%)
            all_authors_field (0.2%)
            authors_no_initials (0.15%)

        TODO: For works that have over 6 authors, we will make two of the following:
            author + field (0.2%)
            two authors + field (0.2%)
            two authors + subfield (0.2%)
            all_authors_field (0.2%)
            all_authors_field_subfield (0.2%)
            authors_no_initials  (1.0%)



        :return:
        """


        print("Creating augmented data...")
        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        df = pl.read_parquet(works_file)

        # Select top 100% of rows
        top_100_percent = df.head(int(df.shape[0] * 1.0))

        # Define probability map for each augmentation type
        augmentation_prob_map = {
            'full_title': 0.05,
            'full_title_field': 0.10,
            'author_field': 0.10,
            'all_authors_field': 0.10,
            'one_author_field_subfield': 0.10,
            'two_authors_field_subfield': 0.10,
            'two_authors_field': 0.15,
            'full_title_field_subfield': 0.02,
            'all_authors_field_subfield': 0.03,
            'field': 0.002,
            'field_subfield': 0.001,
            'trigram_title': 0.10,
            'trigram_title_field': 0.10,
            'trigram_field_subfield': 0.02,
            'authors_no_initials': 0.05,  # Add the new augmentation type with a probability
        }

        def create_augmented_string(row):
            full_string = f"{row['title_string']} {row['authors_string']} {row['field_string']} {row['subfield_string']}"
            author_names = row['author_names']
            title_words = row['title_string'].split()

            # Roll the dice to select an augmentation type
            rand_val = random.random()
            cumulative_prob = 0
            selected_type = None

            for aug_type, prob in augmentation_prob_map.items():
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_type = aug_type
                    break

            augmented_string = ""

            # Create the augmented string based on the selected type
            if selected_type == 'full_title_field_subfield':
                augmented_string = f"{row['title_string']} {row['field_string']} {row['subfield_string']}"
            elif selected_type == 'all_authors_field_subfield':
                augmented_string = f"{' '.join(author_names)} {row['field_string']} {row['subfield_string']}"
            elif selected_type == 'full_title_field':
                augmented_string = f"{row['title_string']} {row['field_string']}"
            elif selected_type == 'full_title':
                augmented_string = f"{row['title_string']}"
            elif selected_type == 'all_authors_field':
                augmented_string = f"{' '.join(author_names)} {row['field_string']}"
            elif selected_type == 'one_author_field_subfield':
                augmented_string = f"{author_names[0] if author_names else ''} {row['field_string']} {row['subfield_string']}"
            elif selected_type == 'field':
                augmented_string = row['field_string']
            elif selected_type == 'field_subfield':
                augmented_string = f"{row['field_string']} {row['subfield_string']}"
            elif selected_type == 'two_authors_field_subfield':
                augmented_string = f"{' '.join(author_names[:2])} {row['field_string']} {row['subfield_string']}"
            elif selected_type == 'two_authors_field':
                augmented_string = f"{' '.join(author_names[:2])} {row['field_string']}"
            elif selected_type == 'author_field':
                augmented_string = f"{author_names[0] if author_names else ''} {row['field_string']}"
            elif selected_type in ['trigram_title', 'trigram_title_field', 'trigram_field_subfield'] and len(
                    title_words) >= 3:
                n = random.randint(2, len(title_words))
                m = random.randint(1, n)
                trigram = ' '.join(title_words[n - m:n])

                if selected_type == 'trigram_title':
                    augmented_string = trigram
                elif selected_type == 'trigram_title_field':
                    augmented_string = f"{trigram} {row['field_string']}"
                elif selected_type == 'trigram_field_subfield':
                    augmented_string = f"{trigram} {row['field_string']} {row['subfield_string']}"
            elif selected_type == 'authors_no_initials':
                # Remove initials from author names
                authors_no_initials = [name for name in author_names if
                                       not (len(name) == 2 and name[1] in ['.', ',']) and not (
                                                   len(name) == 1 and name.isalpha())]
                augmented_string = f"{' '.join(authors_no_initials)} {row['field_string']} {row['subfield_string']}"

            if augmented_string.strip() == full_string.strip():
                # If they're the same, use a random word from full_string
                words = full_string.split()
                if words:
                    augmented_string = random.choice(words)
                else:
                    # If full_string is empty, use a placeholder
                    augmented_string = random.choice(["Science"])

            return {'full_string': full_string, 'augmented_string': augmented_string,
                    'augmentation_type': selected_type}

        # Apply the augmentation to each row
        augmented_df = top_100_percent.with_columns([
            pl.struct(['title_string', 'authors_string', 'field_string', 'subfield_string', 'author_names'])
            .map_elements(create_augmented_string)
            .alias('augmented')
        ]).with_columns([
            pl.col('augmented').struct.field('full_string').alias('full_string_one'),
            pl.col('augmented').struct.field('augmented_string').alias('full_string_two'),
            pl.col('augmented').struct.field('augmentation_type').alias('augmentation_type')
        ]).filter(pl.col('full_string_two') != "")

        # Add additional columns
        augmented_df = augmented_df.with_columns([
            pl.col('work_id').alias('work_id_one'),
            pl.col('work_id').alias('work_id_two'),
            pl.lit('similar').alias('label'),
            pl.lit(1).alias('label_int'),
            pl.lit(0.0).alias('p_value')
        ])

        # Select only the necessary columns
        final_columns = ['work_id_one', 'full_string_one', 'work_id_two', 'full_string_two', 'label', 'label_int',
                         'augmentation_type', 'p_value']
        augmented_df = augmented_df.select(final_columns)

        # Save to parquet file
        output_file = os.path.join(self.datasets_directory, 'works_augmented_data.parquet')
        augmented_df.write_parquet(output_file)

        print(f"Augmented data created and saved to {output_file}")
        print(f"Total augmented pairs: {augmented_df.shape[0]}")

        # Print counts for each augmentation type
        print("\nAugmentation type counts:")
        print(augmented_df.group_by('augmentation_type').count().sort('count', descending=True))

        return augmented_df

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
                    'common_uni_grams': vectorized_unigrams[i],
                    'common_bi_grams': vectorized_bigrams[i],
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
    def restructure_augmented_data(self):

        """

        """

        self.create_augmented_data()

        augmented_data_file = os.path.join(self.datasets_directory, "works_augmented_data.parquet")
        print("Filtering augmented data file...")

        # Read the parquet file
        df = pl.read_parquet(augmented_data_file)

        print("Schema of augmented_data_file:")
        print(df.schema)
        print("\nFirst 20 rows of augmented_data_file:")
        print(df.head(20))

        initial_rows = df.shape[0]
        print(f"Initial number of rows: {initial_rows}")

        # Create a dictionary to keep track of work_id occurrences
        work_id_counter = {}

        # Function to check if a row should be kept
        def keep_row(row):
            work_id_one, work_id_two = row['work_id_one'], row['work_id_two']
            work_id_counter[work_id_one] = work_id_counter.get(work_id_one, 0) + 1
            work_id_counter[work_id_two] = work_id_counter.get(work_id_two, 0) + 1
            return work_id_counter[work_id_one] <= 2 and work_id_counter[work_id_two] <= 2

        # Apply the filtering
        filtered_df = df.filter(pl.struct(['work_id_one', 'work_id_two']).map_elements(keep_row))

        # Reset the counter for the final count
        work_id_counter = {}
        for row in filtered_df.iter_rows(named=True):
            work_id_counter[row['work_id_one']] = work_id_counter.get(row['work_id_one'], 0) + 1
            work_id_counter[row['work_id_two']] = work_id_counter.get(row['work_id_two'], 0) + 1

        def process_common_elements(row):
            # Process full_string_one
            unigrams_one = row['full_string_one'].lower().split()
            bigrams_one = [f"{unigrams_one[i]} {unigrams_one[i + 1]}" for i in range(len(unigrams_one) - 1)]

            # Process full_string_two
            unigrams_two = row['full_string_two'].lower().split()
            bigrams_two = [f"{unigrams_two[i]} {unigrams_two[i + 1]}" for i in range(len(unigrams_two) - 1)]

            # Find common elements
            common_unigrams = list(set(unigrams_one) & set(unigrams_two))
            common_bigrams = list(set(bigrams_one) & set(bigrams_two))

            return pl.Series([common_unigrams, common_bigrams, 1, 1])

        # Apply the processing to each row
        processed_df = filtered_df.with_columns([
            pl.struct(['full_string_one', 'full_string_two'])
            .map_elements(process_common_elements)
            .alias('processed')
        ]).with_columns([
            pl.col('processed').struct.field('column_0').alias('common_uni_grams'),
            pl.col('processed').struct.field('column_1').alias('common_bi_grams'),
            pl.col('processed').struct.field('column_2').cast(pl.Boolean).alias('common_field'),
            pl.col('processed').struct.field('column_3').cast(pl.Boolean).alias('common_subfield')
        ]).drop('processed')

        gc.collect()

        unigrams_df, bigrams_df = self.load_ngrams()

        # Calculate total scores
        insert_data = self.calculate_total_scores(processed_df.to_dicts(), unigrams_df, bigrams_df)

        # Convert insert_data back to DataFrame
        result_df = pl.DataFrame(insert_data)

        result_df = result_df.with_columns(pl.lit('works_augmented_data').alias('source'))

        print("\nFinal schema:")
        print(result_df.schema)
        print("\nFirst 20 rows of final dataframe:")
        print(result_df.head(20))

        final_rows = result_df.shape[0]
        print(f"Final number of rows: {final_rows}")
        print(f"Removed {initial_rows - final_rows} rows")

        # Print work_id occurrence statistics
        print("\nWork ID occurrence statistics:")
        print(f"Number of unique work_ids: {len(work_id_counter)}")
        print(f"Number of work_ids appearing once: {sum(1 for count in work_id_counter.values() if count == 1)}")
        print(f"Number of work_ids appearing twice: {sum(1 for count in work_id_counter.values() if count == 2)}")

        # Save the filtered DataFrame
        result_df.write_parquet(augmented_data_file)
        print(f"Filtered augmented data file saved to {augmented_data_file}")

        return augmented_data_file

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
    def create_sentence_embeddings(self, works_batch_size=100_000):
        works_file = self.works_all_collected_file
        df = pl.read_parquet(works_file)

        total_works = len(df)
        total_batches = (total_works + works_batch_size - 1) // works_batch_size

        model = SentenceTransformer(self.model_path)
        model = DataParallel(model)
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
                embeddings = []
                for i in tqdm(range(0, len(sentences), 64), desc=f"Encoding batch {batch_num + 1}/{total_batches}"):
                    batch = sentences[i:i + 64]
                    batch_embeddings = model(batch).cpu().numpy()
                    embeddings.extend(batch_embeddings)

            batch_data = pl.DataFrame({
                'work_id': work_ids,
                'work_int_id': work_int_ids,
                'work_sentence': sentences,
                'work_embedding': embeddings
            })

            file_name = f'work_embeddings_batch_{batch_num}.parquet'
            file_path = os.path.join(self.embeddings_directory, file_name)
            batch_data.write_parquet(file_path)

            print(f"Processed batch {batch_num + 1}/{total_batches}, saved to {file_path}")

        print(f"Sentence embeddings created and saved in {self.embeddings_directory}")
        print(f"Total works processed: {total_works}")


    def load_ngrams(self):
        unigrams_df = pl.read_parquet(self.unigram_data_file)
        bigrams_df = pl.read_parquet(self.bigram_data_file)
        return unigrams_df, bigrams_df

    def create_sentence_work(self, work_info):
        display_name = work_info.get('title_string', '')
        authors_string = work_info.get('authors_string', '')
        field = work_info.get('field_string', '')
        subfield = work_info.get('subfield_string', '')

        query_string = f"{display_name} {authors_string} {field} {subfield}"
        return query_string


    def sort_files_numerically(self):
        files = os.listdir(self.embeddings_directory)
        parquet_files = [f for f in files if f.endswith('.parquet') and '_embeddings' in f]
        unique_files = list(set(parquet_files))  # Remove duplicates
        sorted_files = sorted(unique_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return sorted_files


    @measure_time
    def build_vector_index(self, output_directory=None, collection_name="Works", N=20_000_000, batch_size=10000, use_gpu=False):
        """
        We will be building this on cpu and then add more vectors later.

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
            file_path = os.path.join(self.vectordb_directory, file)
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
        embeddings = np.array([item['embedding'] for item in all_data])

        print(f"Shape of embeddings: {embeddings.shape}")

        d = embeddings.shape[1]
        n = embeddings.shape[0]
        index_type, nlist, hnsw_m = self.calculate_index_parameters(n)

        print("index_type, nlist, hnsw_m", index_type, nlist, hnsw_m)

        if use_gpu:
            index = self.train_index_gpu(embeddings, d, index_type, nlist, hnsw_m)
        else:
            index = self.train_index_cpu(embeddings, d, index_type, nlist, hnsw_m)

        nlist_num = int(math.sqrt(nlist)) // 2

        nprobe_count = min(512, nlist_num, nlist // 2)
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

        return index_path, mapping_path  # Add this line to return the paths

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
            return "IVF65536,HNSW32,Flat", 65536, 32
        elif 10_000_000 <= collection_size < 100_000_000:
            return "IVF262144,HNSW32,Flat", 262144, 32
        else:  # 100M or more
            return "IVF1048576,HNSW32,PQ16", 1048576, 32

    @measure_time
    def train_index_gpu(self, embeddings, d, index_type, nlist, hnsw_m):
        res = faiss.StandardGpuResources()
        if "HNSW" in index_type:
            quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, 32, 8)
        else:
            index = faiss.index_factory(d, index_type + ",PQ32")
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.train(embeddings)
        gpu_index.add(embeddings)
        return faiss.index_gpu_to_cpu(gpu_index)

    @measure_time
    def train_index_cpu(self, embeddings, d, index_type, nlist, hnsw_m):
        if "HNSW" in index_type:
            quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, 32, 8)
        else:
            index = faiss.index_factory(d, index_type + ",PQ32")
        index.train(embeddings)
        index.add(embeddings)
        return index


    @measure_time
    def generate_training_pairs(self, batch_size=512, initial_k=128, distance_threshold=0.1):
        """
        TODO: When we look at pairs here, when we compute distances of every single pair, we want to store those
            distances in a dataframe and parquet file, for later use.
            So, we actually want the following:

        dataframe: [work_id_one, work_id_two, distance, total_score]


        :param batch_size:
        :param initial_k:
        :return:
        """


        self.load_index_and_mapping()
        self.load_works_data()
        unigrams_df, bigrams_df = self.load_ngrams()

        works_filtered_df = pl.read_parquet(self.works_all_collected_file)

        pairs_generated = 0
        processed_works = set()

        index_path = self.index_works_file
        index = faiss.read_index(index_path)

        mapping_path = self.id_mapping_works_file
        mapping_df = pl.read_feather(mapping_path)

        print("Columns in the mapping DataFrame:")
        print(mapping_df.columns)

        faiss_to_works_id = dict(zip(mapping_df['faiss_index'], mapping_df['Works_ids']))

        cited_by_count_map = dict(zip(works_filtered_df['work_id'], works_filtered_df['cited_by_count']))

        unigrams_dict = self.works_df['unigrams'].to_dict()

        thresholds = [sum(self.num_knn_pairs // (2 ** (i + 0)) for i in range(1, j + 1)) for j in range(1, 10)]

        k = initial_k

        threshold_index = 0

        max_batch_size = 4096 * 32  # Maximum batch size
        batch_size = min(batch_size, max_batch_size)  # Ensure initial batch size doesn't exceed max

        while pairs_generated < (self.num_knn_pairs * 2.0):
            # Dynamically adjust k
            if threshold_index < len(thresholds) and pairs_generated >= thresholds[threshold_index]:
                k = max(k // 2, 8)  # Reduce k by half, whenever our collected data grows by another half.
                threshold_index += 1
                batch_size = min(batch_size * 2, max_batch_size)  # Double batch size, but cap it at max_batch_size

                print(f"Adjusting k to {k} and batch size to {batch_size}")

            unprocessed_work_ids = self.works_df[
                                       (self.works_df['work_id_search_count'] == 0) &
                                       (~self.works_df.index.is_in(processed_works))
                                       ].index[:batch_size].tolist()

            if not unprocessed_work_ids:
                print("No more unprocessed works found.")
                break

            similar_works_df = self.batch_search_similar_works(unprocessed_work_ids, k, index, faiss_to_works_id,
                                                               distance_threshold)

            all_pairs = []
            all_distances = []
            work_pair_count = {}
            print("Length of processed works: ", len(processed_works))
            gc.collect()

            # In the main loop
            for query_work_id in tqdm(unprocessed_work_ids, desc="Processing work IDs"):
                similar_works = similar_works_df[similar_works_df['query_work_id'] == query_work_id]
                similar_work_ids = similar_works['similar_work_id'].to_list()
                distances = similar_works['distance'].to_list()

                valid_pairs, counts, new_work_pair_count = self.filter_and_count_pairs(similar_work_ids, unigrams_dict,
                                                                                       self.work_details, k)

                all_pairs.extend(valid_pairs)
                all_distances.extend([distances[similar_work_ids.index(pair[1])] for pair in valid_pairs])

                # Update the global work_pair_count
                for work_id, count in new_work_pair_count.items():
                    work_pair_count[work_id] = work_pair_count.get(work_id, 0) + count

            if k > 128:
                min_count = 3
                max_appearances = 8
            elif 128 >= k > 64:
                min_count = 2
                max_appearances = 6
            else:
                min_count = 2
                max_appearances = 4

            filtered_pairs, filtered_distances = self.filter_pairs_by_count(all_pairs, all_distances, work_pair_count,
                                                                            cited_by_count_map, min_count=min_count)
            print(f"Total pairs after filtering for min_count req of {min_count} or more: {len(filtered_pairs)}")

            all_pairs, all_distances = self.filter_pairs_by_appearance(filtered_pairs, filtered_distances,
                                                                       cited_by_count_map,
                                                                       max_appearances=max_appearances)

            print(f"Total pairs after filtering out max_appearances counts over {max_appearances}: {len(all_pairs)}")

            work_ids = set([work_id for pair in all_pairs for work_id in pair])

            work_details = self.fetch_work_details(work_ids, works_filtered_df)

            vectorized_unigrams, vectorized_bigrams, vectorized_fields, vectorized_subfields = self.process_and_vectorize_common_elements(
                work_details, all_pairs)

            insert_data = self.create_insert_data(all_pairs, work_details, vectorized_unigrams, vectorized_bigrams,
                                                  vectorized_fields, vectorized_subfields, all_distances)
            insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)
            insert_data = self.process_p_values(insert_data)

            # TODO:

            processed_works.update(unprocessed_work_ids)
            processed_works.update(work_ids)

            self.update_processed_works(unprocessed_work_ids, work_ids)

            self.batch_insert_siamese_data(insert_data)

            pairs_generated += len(insert_data)
            print(f"Generated {pairs_generated} pairs so far. Current k: {k}")

            if (pairs_generated >= (self.num_knn_pairs * 2.0)) or len(processed_works) > int(len(self.works_df) * 0.99):
                break

        self.save_processed_data()

        print(f"Total pairs generated: {pairs_generated}")


    def filter_and_count_pairs(self, similar_works, unigrams_dict, work_details, k):
        """
        I wish to modify this method so that we create hashmap that does the following.
        The hashmap will put pairs of work1_id, work2_id in it and say whether it
        has common_3 condition satisified, common_2, common_1, or common_field_subfield.
        Now, if we have get a pair of work_id's where both work_id's have been seen before,
        and it only has common_1 o common_field_subfield,
        then we do not add the pair-we skip over it.

        We shall need to make a dictionary mapping work_id to counts for common_3


        :param similar_works:
        :param unigrams_dict:
        :param work_details:
        :param k:
        :return:
        """

        common_stop_words = self.get_stop_words()
        possible_pairs = list(combinations(similar_works, 2))
        random_numbers = np.random.random(len(possible_pairs))

        valid_pairs = []
        counts = {"common_3": 0, "common_2": 0, "common_1": 0, "common_field_subfield": 0}
        work_pair_count = {}

        for idx, (work1_id, work2_id) in enumerate(possible_pairs):
            work1 = work_details.get(work1_id, {})
            work2 = work_details.get(work2_id, {})

            work1_unigrams = set(unigrams_dict.get(work1_id, [])) - common_stop_words
            work2_unigrams = set(unigrams_dict.get(work2_id, [])) - common_stop_words

            common_unigrams_count = len(work1_unigrams & work2_unigrams)
            common_field = work1.get('field_string') == work2.get('field_string')

            rand_num = random_numbers[idx]
            is_valid = False

            if common_unigrams_count >= 3:
                counts["common_3"] += 1
                is_valid = True
            elif common_unigrams_count >= 2 and rand_num > 0.5:
                counts["common_2"] += 1
                is_valid = True
            elif k < 150 and (common_unigrams_count >= 2):
                is_valid = True
            elif k < 100 and (common_unigrams_count >= 1 or common_field or rand_num > 0.98):
                is_valid = True
            elif k < 50 and (common_unigrams_count >= 1 or common_field or rand_num > 0.95):
                is_valid = True
            elif common_unigrams_count >= 1 and rand_num > 0.9:
                counts["common_1"] += 1
                is_valid = True
            elif common_field and rand_num > 0.9:
                counts["common_field_subfield"] += 1
                is_valid = True
            elif rand_num > 0.9999:
                is_valid = True
            if is_valid:
                valid_pairs.append((work1_id, work2_id))
                work_pair_count[work1_id] = work_pair_count.get(work1_id, 0) + 1
                work_pair_count[work2_id] = work_pair_count.get(work2_id, 0) + 1

        return valid_pairs, counts, work_pair_count

    @measure_time
    def filter_pairs_by_count(self, all_pairs, all_distances, work_pair_count, cited_by_count_map, min_count=4):
        # Filter pairs and distances based on minimum count
        filtered_pairs_and_distances = [
            (pair, distance) for pair, distance in zip(all_pairs, all_distances)
            if work_pair_count.get(pair[0], 0) >= min_count and work_pair_count.get(pair[1], 0) >= min_count
        ]

        # Sort pairs by combined cited_by_count in descending order
        sorted_pairs_and_distances = sorted(
            filtered_pairs_and_distances,
            key=lambda x: (cited_by_count_map.get(x[0][0], 0) + cited_by_count_map.get(x[0][1], 0)),
            reverse=True
        )

        # Separate the sorted pairs and distances
        sorted_pairs, sorted_distances = zip(*sorted_pairs_and_distances) if sorted_pairs_and_distances else ([], [])

        return list(sorted_pairs), list(sorted_distances)

    @measure_time
    def filter_pairs_by_appearance(self, filtered_pairs, filtered_distances, cited_by_count_map, max_appearances=6):
        final_pairs = []
        final_distances = []
        work_appearance_count = {}

        # Sort pairs and distances by combined cited_by_count in descending order
        sorted_pairs_and_distances = sorted(
            zip(filtered_pairs, filtered_distances),
            key=lambda x: (cited_by_count_map.get(x[0][0], 0) + cited_by_count_map.get(x[0][1], 0)),
            reverse=True
        )

        for pair, distance in sorted_pairs_and_distances:
            if work_appearance_count.get(pair[0], 0) < max_appearances and \
                    work_appearance_count.get(pair[1], 0) < max_appearances:
                final_pairs.append(pair)
                final_distances.append(distance)
                work_appearance_count[pair[0]] = work_appearance_count.get(pair[0], 0) + 1
                work_appearance_count[pair[1]] = work_appearance_count.get(pair[1], 0) + 1

        return final_pairs, final_distances

    @measure_time
    def process_p_values(self, insert_data):
        scores, mean_score, median_score, std_score = self.calculate_score_statistics(insert_data)
        insert_data = self.assign_p_values(insert_data, mean_score, median_score, std_score)
        filtered_data = self.filter_by_p_value(insert_data)

        work_id_count = self.create_work_id_count(filtered_data)
        final_data = self.remove_single_occurrence_pairs(filtered_data, work_id_count)
        self.print_p_value_statistics(final_data)
        return final_data


    @measure_time
    def update_processed_works(self, queried_work_ids, found_work_ids):
        # Combine queried and found work_ids, removing duplicates
        all_work_ids = set(queried_work_ids) | set(found_work_ids)

        # Update work_id_search_count in memory
        for work_id in all_work_ids:
            self.work_id_search_count[work_id] = self.work_id_search_count.get(work_id, 0) + 1

        # Update work_id_search_count in the DataFrame
        self.works_df.loc[self.works_df.index.is_in(all_work_ids), 'work_id_search_count'] += 1

        print(f"Updated work_id_search_count for {len(all_work_ids)} works")

    @measure_time
    def calculate_total_scores(self, insert_data, unigrams_df, bigrams_df):
        """
        Calculate total scores using Polars, with modifications as per TODO comments. We need to make test vectorizated gram scores because loading up the dictionary in this method takes a long time.
        """
        # Convert insert_data to a Polars DataFrame if it's not already
        if not isinstance(insert_data, pl.DataFrame):
            df = pl.DataFrame(insert_data)
        else:
            df = insert_data

        # Calculate gram scores
        df = df.with_columns([
            self.vectorized_gram_scores('common_uni_grams', unigrams_df, testing_method=True).alias('unigram_score'),
            self.vectorized_gram_scores('common_bi_grams', bigrams_df, testing_method=True).alias('bigram_score')
        ])

        # Calculate sum of gram scores instead of average
        df = df.with_columns([
            (pl.col('unigram_score') + pl.col('bigram_score')).alias('sum_gram_score')
        ])

        scalar_multiplier = 0.05
        df = df.with_columns([
            (pl.when(pl.col('common_field') >= 0)
             .then(pl.col('common_field') * (3.0 + 2.0 * scalar_multiplier * pl.col('sum_gram_score')))
             .otherwise(0)).alias('field_score'),
            (pl.when(pl.col('common_subfield') >= 0)
             .then(pl.col('common_subfield') * (1.0 + scalar_multiplier * pl.col('sum_gram_score')))
             .otherwise(0)).alias('subfield_score')
        ])

        # Calculate total score
        df = df.with_columns([
            (pl.col('unigram_score') + pl.col('bigram_score') + pl.col('field_score') + pl.col('subfield_score')).alias(
                'total_score')
        ])

        # Convert back to list of dictionaries
        return df.to_dicts()

    @measure_time
    def vectorized_gram_scores(self, gram_column, gram_df, testing_method=False):
        """
        Calculate vectorized gram scores using Polars.
        If testing_method is True, return random scores instead of actual calculations.
        """
        if testing_method:
            # Define a function to return a random score between 0 and 5
            def calculate_random_score(gram_list):
                return random.uniform(0, 5) * len(gram_list)

            # Use pl.col().map() to apply the random score function to each list in the column
            return pl.col(gram_column).map_elements(lambda x: calculate_random_score(x))
        else:
            gram_type = "unigram_type" if 'unigram_type' in gram_df.columns else "bigram_type"

            # Create a dictionary of gram scores
            scores_dict = dict(zip(gram_df[gram_type], gram_df['score']))

            # Define a function to calculate the score for a list of grams
            def calculate_score(gram_list):
                return sum(scores_dict.get(gram, 2.5) for gram in gram_list)

            # Use pl.col().map() to apply the function to each list in the column
            return pl.col(gram_column).map_elements(lambda x: calculate_score(x))


    @measure_time
    def calculate_score_statistics(self, insert_data):
        scores = [item['total_score'] for item in insert_data]
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        return scores, mean_score, median_score, std_score


    @measure_time
    def assign_p_values(self, insert_data, mean_score, median_score, std_score):
        # Convert insert_data to a DataFrame if it's not already
        if not isinstance(insert_data, pl.DataFrame):
            df = pl.DataFrame(insert_data)
        else:
            df = insert_data.clone()

        # Vectorized calculation of z_scores and p_values
        df = df.with_columns([
            ((pl.col('total_score') - mean_score) / std_score).alias('z_score'),
            (1 - pl.col('z_score').map_elements(norm.cdf)).alias('p_value')
        ])

        # If the original input was a list of dictionaries, convert back
        if not isinstance(insert_data, pl.DataFrame):
            return df.to_dicts()
        else:
            return df

    @measure_time
    def filter_by_p_value(self, insert_data):
        if isinstance(insert_data, pl.DataFrame):
            return insert_data.filter((pl.col('p_value') <= 0.49) | (pl.col('p_value') >= 0.51))
        elif isinstance(insert_data, list):
            df = pl.DataFrame(insert_data)
            filtered_df = df.filter((pl.col('p_value') <= 0.49) | (pl.col('p_value') >= 0.51))
            return filtered_df.to_dicts()
        else:
            raise TypeError("insert_data must be a DataFrame or a list of dictionaries")

    @measure_time
    def remove_single_occurrence_pairs(self, filtered_data, work_id_count):
        if not isinstance(filtered_data, pl.DataFrame):
            df = pl.DataFrame(filtered_data)
        else:
            df = filtered_data

        def has_both_occurrences(work_id):
            return work_id_count[work_id]['above'] > 0 and work_id_count[work_id]['below'] > 0

        filtered_df = df.filter(
            pl.col('work_id_one').map_elements(has_both_occurrences) &
            pl.col('work_id_two').map_elements(has_both_occurrences)
        )

        if not isinstance(filtered_data, pl.DataFrame):
            return filtered_df.to_dicts()
        else:
            return filtered_df

    @measure_time
    def create_work_id_count(self, filtered_data):
        if not isinstance(filtered_data, pl.DataFrame):
            df = pl.DataFrame(filtered_data)
        else:
            df = filtered_data

        work_id_count = {}

        for row in df.iter_rows(named=True):
            for work_id in [row['work_id_one'], row['work_id_two']]:
                if work_id not in work_id_count:
                    work_id_count[work_id] = {'above': 0, 'below': 0}
                if row['p_value'] > 0.5:
                    work_id_count[work_id]['above'] += 1
                else:
                    work_id_count[work_id]['below'] += 1

        return work_id_count


    @measure_time
    def print_p_value_statistics(self, final_data):
        p_values = [item['p_value'] for item in final_data]
        print(f"P-value at 0.4: {np.percentile(p_values, 40):.4f}")
        print(f"P-value at 0.6: {np.percentile(p_values, 60):.4f}")
        print(f"Lower quartile (25th percentile): {np.percentile(p_values, 25):.4f}")
        print(f"Upper quartile (75th percentile): {np.percentile(p_values, 75):.4f}")

    @measure_time
    def load_index_and_mapping(self):
        index_path = os.path.join(self.output_directory, "index_works.bin")
        self.vector_index = faiss.read_index(index_path)
        mapping_path = os.path.join(self.output_directory, "id_mapping_works.parquet")
        self.faiss_to_work_id_mapping = pl.read_feather(mapping_path)

    @measure_time
    def load_works_data(self, duplicates_check=True):
        self.works_df = pl.read_parquet(self.works_all_collected_file)

        if 'work_id' not in self.works_df.columns:
            print("Warning: 'work_id' column not found. Creating it from the index.")
            self.works_df['work_id'] = self.works_df.index

        if duplicates_check:
            # Check for duplicate work_ids
            duplicates = self.works_df['work_id'].duplicated()
            if duplicates.any():
                print(f"Found {duplicates.sum()} duplicate work_ids. Keeping the first occurrence of each.")
                self.works_df = self.works_df.loc[~self.works_df['work_id'].duplicated(keep='first')]

                # Save the filtered DataFrame back to the parquet file
                self.works_df.to_parquet(self.works_all_collected_file, index=False)
                print(
                    f"Saved filtered DataFrame with {len(self.works_df)} unique works to {self.works_all_collected_file}")

        # Set 'work_id' as index while keeping it as a column
        self.works_df.set_index('work_id', inplace=True, drop=False)
        self.work_details = self.works_df.to_dict('index')

        print(f"Loaded {len(self.works_df)} unique works.")

    def get_stop_words(self):
        # You can expand this set of stop words as needed
        return {
            'a', 'A', 'about', 'About', 'above', 'Above', 'after', 'After', 'again', 'Again', 'against', 'Against',
            'all', 'All', 'am', 'Am', 'an', 'An', 'and', 'And', 'any', 'Any', 'are', 'Are', 'as', 'As', 'at', 'At',
            'be', 'Be', 'because', 'Because', 'been', 'Been', 'before', 'Before', 'being', 'Being', 'below', 'Below',
            'between', 'Between', 'both', 'Both', 'but', 'But', 'by', 'By', 'can', 'Can', 'did', 'Did', 'do', 'Do',
            'does', 'Does', 'doing', 'Doing', 'down', 'Down', 'during', 'During', 'each', 'Each', 'few', 'Few',
            'for', 'For', 'from', 'From', 'further', 'Further', 'had', 'Had', 'has', 'Has', 'have', 'Have',
            'having', 'Having', 'he', 'He', 'her', 'Her', 'here', 'Here', 'hers', 'Hers', 'herself', 'Herself',
            'him', 'Him', 'himself', 'Himself', 'his', 'His', 'how', 'How', 'i', 'I', 'if', 'If', 'in', 'In',
            'into', 'Into', 'is', 'Is', 'it', 'It', 'its', 'Its', 'itself', 'Itself', 'just', 'Just', 'me', 'Me',
            'more', 'More', 'most', 'Most', 'my', 'My', 'myself', 'Myself', 'no', 'No', 'nor', 'Nor', 'not', 'Not',
            'now', 'Now', 'of', 'Of', 'off', 'Off', 'on', 'On', 'once', 'Once', 'only', 'Only', 'or', 'Or',
            'other', 'Other', 'our', 'Our', 'ours', 'Ours', 'ourselves', 'Ourselves', 'out', 'Out', 'over', 'Over',
            'own', 'Own', 'same', 'Same', 'she', 'She', 'should', 'Should', 'so', 'So', 'some', 'Some', 'such', 'Such',
            'than', 'Than', 'that', 'That', 'the', 'The', 'their', 'Their', 'theirs', 'Theirs', 'them', 'Them',
            'themselves', 'Themselves', 'then', 'Then', 'there', 'There', 'these', 'These', 'they', 'They',
            'this', 'This', 'those', 'Those', 'through', 'Through', 'to', 'To', 'too', 'Too', 'under', 'Under',
            'until', 'Until', 'up', 'Up', 'very', 'Very', 'was', 'Was', 'we', 'We', 'were', 'Were', 'what', 'What',
            'when', 'When', 'where', 'Where', 'which', 'Which', 'while', 'While', 'who', 'Who', 'whom', 'Whom',
            'why', 'Why', 'with', 'With', 'would', 'Would', 'you', 'You', 'your', 'Your', 'yours', 'Yours',
            'yourself', 'Yourself', 'yourselves', 'Yourselves', ',', '.', ':', ';', '!', '?', '"',
            "'", '(', ')', '[', ']', '{', '}', '-', '–', '—', '/', '|', '@', '#',
            '$', '%', '^', '&', '*', '+', '=', '<', '>', '`', '~'
        }


    @measure_time
    def batch_insert_siamese_data(self, insert_data):
        # Append new data to the in-memory list
        self.works_knn_search.extend(insert_data)
        print(f"Added {len(insert_data)} new entries to works_knn_search")


    @measure_time
    def save_processed_data(self):
        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")

        try:
            self.works_df.write_parquet(works_file)
        except Exception as e:
            print("Error saving works_df:", e)

        print(f"Updated work_id_search_count for {len(self.work_id_search_count)} works in Parquet file")

        siamese_file = os.path.join(self.datasets_directory, "works_knn_search.parquet")

        # Save siamese data
        if not os.path.exists(siamese_file):
            columns = [
                'work_id_one', 'full_string_one', 'work_id_two', 'full_string_two',
                'common_uni_grams', 'common_bi_grams', 'common_field', 'common_subfield',
                'total_score', 'label', 'label_int', 'p_value', 'distance'  # Add 'distance' to the columns
            ]
            df = pl.DataFrame(self.works_knn_search, schema=columns)
        else:
            existing_df = pl.read_parquet(siamese_file)
            new_df = pl.DataFrame(self.works_knn_search)
            df = pl.concat([existing_df, new_df])

        df.write_parquet(siamese_file)
        print(f"Saved {len(self.works_knn_search)} entries to works_knn_search data Parquet file")

        # Clear the in-memory data
        self.work_id_search_count.clear()
        self.works_knn_search.clear()

        # Force garbage collection
        gc.collect()

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
    def batch_search_similar_works(self, work_ids, k, index, faiss_to_works_id, distance_threshold=0.1):
        work_embeddings = self.batch_encode_works(
            [self.create_sentence_work(self.work_details[work_id]) for work_id in work_ids])

        # Perform the initial search
        distances, indices = self.perform_batch_search(index, work_embeddings, k)

        # Compute pairwise distances for the retrieved vectors
        pairwise_distances = self.compute_pairwise_distances(work_embeddings)

        results = []
        for i, work_id in enumerate(work_ids):
            filtered_indices = []
            filtered_distances = []

            for j in range(k):
                faiss_idx = int(indices[i][j])
                if faiss_idx not in filtered_indices:  # Check if this index is already filtered
                    try:
                        similar_work_id = faiss_to_works_id[faiss_idx]

                        # Check pairwise distance
                        if j > 0 and np.min(pairwise_distances[i, filtered_indices]) < distance_threshold:
                            continue  # Skip this result if it's too close to previously added results

                        filtered_indices.append(j)
                        filtered_distances.append(distances[i][j])

                        results.append({
                            'query_work_id': work_id,
                            'similar_work_id': similar_work_id,
                            'distance': float(distances[i][j])
                        })
                    except KeyError:
                        print(f"Warning: No mapping found for FAISS index {faiss_idx}")

        return pl.DataFrame(results)

    @measure_time
    def compute_pairwise_distances(self, vectors):
        distances = pdist(vectors)
        distance_matrix = squareform(distances)
        return distance_matrix

    @measure_time
    def fetch_work_details(self, work_ids, works_filtered_df, truncated=False, filter_works=True):
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
    def create_common_title_works(self):
        print("Creating common_title_works.parquet file...")

        # Load the works_all_collected.parquet file and create a hashmap of work_id to title_string
        works_df = pl.read_parquet(self.works_all_collected_file)
        work_id_to_title = dict(zip(works_df['work_id'], works_df['title_string']))

        # Get the set of stop words
        stop_words = self.get_stop_words()

        # Initialize list to store common title pairs
        self.common_title_pairs = []

        self.process_file_for_common_titles(self.works_common_authors_file, work_id_to_title, stop_words)
        self.process_file_for_common_titles(self.works_augmented_data_file, work_id_to_title, stop_words)
        self.process_file_for_common_titles(self.works_knn_search_file, work_id_to_title, stop_words)

        # Create a DataFrame from the common title pairs
        common_title_df = pl.DataFrame(self.common_title_pairs)

        # Save the DataFrame as a parquet file
        common_title_df.write_parquet(self.works_common_titles_file)
        print(f"Created common_title_works.parquet with {len(common_title_df)} pairs")

        self.common_title_pairs = []
        gc.collect()

    def process_file_for_common_titles(self, file_path, work_id_to_title, stop_words):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        df = pl.read_parquet(file_path)

        def process_row(row):
            work_id_one = row['work_id_one']
            work_id_two = row['work_id_two']
            title_one = work_id_to_title.get(work_id_one, "")
            title_two = work_id_to_title.get(work_id_two, "")

            # Get title unigrams for both works
            title_unigrams_one = set(title_one.lower().split()) - stop_words
            title_unigrams_two = set(title_two.lower().split()) - stop_words

            # Find common title unigrams
            common_title_unigrams = title_unigrams_one.intersection(title_unigrams_two)
            if len(common_title_unigrams) < 3:
                return None

            # Get title bigrams for both works
            title_bigrams_one = set([f"{title_one.lower().split()[i]} {title_one.lower().split()[i + 1]}" for i in
                                     range(len(title_one.split()) - 1)])
            title_bigrams_two = set([f"{title_two.lower().split()[i]} {title_two.lower().split()[i + 1]}" for i in
                                     range(len(title_two.split()) - 1)])

            # Find common title bigrams
            common_title_bigrams = title_bigrams_one.intersection(title_bigrams_two)

            # Get title trigrams for both works
            title_trigrams_one = set(
                [f"{title_one.lower().split()[i]} {title_one.lower().split()[i + 1]} {title_one.lower().split()[i + 2]}"
                 for i in range(len(title_one.split()) - 2)])
            title_trigrams_two = set(
                [f"{title_two.lower().split()[i]} {title_two.lower().split()[i + 1]} {title_two.lower().split()[i + 2]}"
                 for i in range(len(title_two.split()) - 2)])

            # Find common title trigrams
            common_title_trigrams = title_trigrams_one.intersection(title_trigrams_two)

            # Calculate the threshold for unigrams
            unigram_threshold = 3 + math.ceil(min(len(title_unigrams_one), len(title_unigrams_two)) / 3)

            # Check if the pair meets the refined conditions
            if (len(common_title_unigrams) >= unigram_threshold or len(common_title_bigrams) >= 2 or len(
                    common_title_trigrams) >= 1):
                return {
                    'work_id_one': work_id_one,
                    'work_id_two': work_id_two,
                    'common_title_unigrams': list(common_title_unigrams),
                    'common_title_bigrams': list(common_title_bigrams),
                    'common_title_trigrams': list(common_title_trigrams),
                    'total_score': row['total_score'],
                    'source': "common_title_works",
                }
            return None

        # Process rows and collect results
        results = []
        for row in tqdm(df.iter_rows(named=True), desc=f"Processing {os.path.basename(file_path)}", total=df.shape[0]):
            result = process_row(row)
            if result:
                results.append(result)

        # Extend common_title_pairs with the results
        self.common_title_pairs.extend(results)

        # Force garbage collection
        gc.collect()

    @measure_time
    def remove_duplicate_pairs(self):
        print("Removing duplicate pairs from selected files...")

        files_to_deduplicate = [
            self.works_common_authors_file,
            self.works_common_titles_file,
            self.works_knn_search_file
        ]

        all_pairs = set()
        deduplicated_data = []

        for file in files_to_deduplicate:
            if os.path.exists(file):
                df = pl.read_parquet(file)
                for row in df.iter_rows(named=True):
                    pair = (row['work_id_one'], row['work_id_two'])
                    if pair not in all_pairs:
                        all_pairs.add(pair)
                        deduplicated_data.append(row)

        # Convert deduplicated_data back to a DataFrame
        deduplicated_df = pl.DataFrame(deduplicated_data)

        # Save deduplicated data to a new file
        deduplicated_file = os.path.join(self.datasets_directory, "deduplicated_pairs.parquet")
        deduplicated_df.write_parquet(deduplicated_file)

        print(f"Removed duplicates. Saved {len(deduplicated_df)} unique pairs to {deduplicated_file}")
        return deduplicated_file

    @measure_time
    def generate_all_work_id_pairs_dataset(self, sort_by_distance=True):
        print("Generating all work ID pairs dataset...")

        # First, remove duplicates from specified files
        deduplicated_file = self.remove_duplicate_pairs()

        files = [
            deduplicated_file,
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

        output_file = self.datasets_directory
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


def setup_directories(environment='local'):
    # TODO: make sure the directory works for linux env.

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
        'preprocess_and_calculate_ngrams': False,
        'batch_update_ngram_scores': False,
        'create_sentence_embeddings': False,
        'calculate_density_scores': False,
        'build_vector_index': False,
        'generate_training_pairs': False,
        'create_common_title_works': False,
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
        num_knn_pairs=300_000,
        num_works_collected=300_000,
        mongo_url="mongodb://localhost:27017/",
        mongo_database_name="OpenAlex",
        mongo_works_collection_name="Works"
    )

    encoder.run()
    encoder.triplets_quality_control_statistics()


