import gc
import glob
import json
import math
import os
import random
import time
from collections import Counter
from itertools import combinations
import psutil
import cupy as cp
import faiss
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch
from pymongo import MongoClient
from scipy.stats import norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from torch.nn.parallel import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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


class DatasetConstructionSentenceEncoder:
    """

    We will be adjusting this class so it constructs three kinds of triplet datasets.
    One shall be of the variant where we make this string:

    full_string = f"{row['title_string']} {row['authors_string']} {row['field_string']} {row['subfield_string']}"

    Another shall be of the variant where we make this string:

    full_string = f"{row['title_string']} {row['authors_string']} {row['field_string']} {row['subfield_string']} {row['keyphrases_string']}"

    Another shall be of the variant where we make this string:

    full_string = f"{row['field_string']} {row['subfield_string']} {row['topic_string']} {row['keyphrases_string']}"

    For now, I wish to focus on this one:

    full_string = f"{row['title_string']} {row['authors_string']} {row['field_string']} {row['subfield_string']}"

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
        To do this, we retrieve the similarities when we do knn vector retrieval, and compute them.
        Then later on, for hard negative mining, we select the hard negative that is highest similarity score to the anchor and positive, without
        going below the 0.5% and 1% thresholds.








    """

    def __init__(self,
                 model_path,
                 output_directory,
                 datasets_directory,
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
        self.output_dir = output_directory
        self.input_directory = r'workspace'
        self.output_directory = output_directory



        self.run_params = run_params

        # Directory structure
        self.workspace_dir = "/workspace"
        self.datasets_dir = os.path.join(self.workspace_dir, "datasets")
        self.embeddings_dir = os.path.join(self.workspace_dir, "embeddings")
        self.output_dir = os.path.join(self.workspace_dir, "output")
        self.embeddings_output_directory = os.path.join(self.workspace_dir, "output", "embeddings")

        # Create directories
        for directory in [self.datasets_dir, self.embeddings_dir, self.output_dir, self.embeddings_output_directory]:
            os.makedirs(directory, exist_ok=True)

        # File paths
        self.works_all_collected_file = os.path.join(self.datasets_dir, "works_all_collected.parquet")
        self.works_common_authors_file = os.path.join(self.datasets_dir, "works_common_authors.parquet")
        self.works_common_authors_filtered_file = os.path.join(self.datasets_dir, "works_common_authors_filtered.parquet")
        self.works_common_titles_file = os.path.join(self.datasets_dir, "common_title_works.parquet")
        self.works_knn_search_file = os.path.join(self.datasets_dir, "works_knn_search.parquet")
        self.softer_negatives_pool_file = os.path.join(self.datasets_dir, "hard_negatives_pool.parquet")
        self.works_augmented_data_file = os.path.join(self.datasets_dir, "works_augmented_data.parquet")
        self.triplet_work_ids_only_file = os.path.join(self.datasets_dir, "triplet_work_ids_only.parquet")
        self.id_mapping_works_file = os.path.join(self.datasets_dir, "id_mapping_works.parquet")
        self.index_works_file = os.path.join(self.datasets_dir, "index_works.bin")
        self.triplets_file = os.path.join(self.datasets_dir, "triplets.parquet")
        self.unigram_data_file = os.path.join(self.output_dir, "unigram_data.parquet")
        self.bigram_data_file = os.path.join(self.output_dir, "bigram_data.parquet")

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Other initializations
        self.work_id_search_count = {}
        self.works_knn_search = []
        self.vector_index = None
        self.faiss_to_work_id_mapping = None
        self.works_df = None
        self.work_details = {}

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
            self.create_sentence_embeddings(works_batch_size=100000)

        if self.run_params.get('build_vector_index', False):
            self.build_vector_index()

        if self.run_params.get('generate_training_pairs', False):
            self.generate_training_pairs(batch_size=4096, initial_k=128)

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
            self.softer_negatives_pool_file,
            self.triplet_work_ids_only_file
        ]

        for file in parquet_files:
            if os.path.exists(file):
                df = pd.read_parquet(file)
                print(f"\nFile: {os.path.basename(file)}")
                print("Schema:")
                print(df.dtypes.to_string())
                print("\nHead (50 rows):")
                print(df.head(50).to_string())
                print("\nTail (50 rows):")
                print(df.tail(50).to_string())
                # Force garbage collection
                del df
                gc.collect()
            else:
                print(f"File not found: {file}")



    @measure_time
    def collect_all_works_metadata(self, abstract_include=False):
        """
        Collects metadata for all works in a single pass.
        """
        self.establish_mongodb_connection()

        print("Collecting metadata for all works...")
        output_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        common_authors_file = os.path.join(self.datasets_directory, "works_common_authors.parquet")

        author_work_map = {}
        total_processed = 0
        new_rows = []
        common_author_pairs = []

        cursor = self.mongodb_works_collection.find().sort("works_int_id", 1)

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

                    if author['id'] in author_work_map:
                        common_author_pairs.append((author_work_map[author['id']], work_id))
                    author_work_map[author['id']] = work_id

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
                'author_names': author_names,  # Store the full list of author names
                'field_string': field,
                'subfield_string': subfield,
                'abstract_string': abstract_string,
                'unigrams': unigrams,
                'bigrams': bigrams,
                'cited_by_count': cited_by_count
            })

            total_processed += 1

            if total_processed % 100000 == 0:
                print(f"Processed {total_processed} works")

            if total_processed >= self.num_works_collected:
                break

        self.close_mongodb_connection()

        # Create and save the main DataFrame
        works_df = pd.DataFrame(new_rows)
        works_df.to_parquet(output_file, index=False)
        print(f"Saved {len(works_df)} works to {output_file}")

        # Create and save the common authors DataFrame
        common_authors_df = pd.DataFrame(common_author_pairs, columns=['work_id_one', 'work_id_two'])
        common_authors_df.to_parquet(common_authors_file, index=False)
        print(f"Saved {len(common_authors_df)} common author pairs to {common_authors_file}")

        print(f"Total works processed: {total_processed}")
        print(f"Total unique author IDs: {len(author_work_map)}")

    @measure_time
    def restructure_common_authors(self):
        """
        We may rework this. And we may not want to save over works_common_authors.parquet here.
        self.works_common_authors_file = os.path.join(self.datasets_directory, "works_common_authors.parquet")
        :return:
        """

        common_authors_file = os.path.join(self.datasets_directory, "works_common_authors.parquet")
        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")

        print("Filtering common authors file...")

        # Read the parquet files
        df = pd.read_parquet(common_authors_file)
        works_df = pd.read_parquet(works_file)

        initial_rows = len(df)
        print(f"Initial number of rows: {initial_rows}")

        # Create a dictionary mapping work_id to cited_by_count
        cited_by_count_dict = dict(zip(works_df['work_id'], works_df['cited_by_count']))

        # Add cited_by_count for work_id_one and work_id_two
        df['cited_by_count_one'] = df['work_id_one'].map(cited_by_count_dict)
        df['cited_by_count_two'] = df['work_id_two'].map(cited_by_count_dict)

        # Calculate combined cited_by_count
        df['combined_cited_by_count'] = df['cited_by_count_one'] + df['cited_by_count_two']
        df = df.sort_values('combined_cited_by_count', ascending=False)

        print(df[['work_id_one', 'work_id_two', 'cited_by_count_one', 'cited_by_count_two',
                  'combined_cited_by_count']].head())

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
        filtered_df = df[df.apply(keep_row, axis=1)]

        # Fetch work details
        all_work_ids = set(filtered_df['work_id_one']) | set(filtered_df['work_id_two'])
        work_details = self.fetch_work_details(all_work_ids, works_df, truncated=False, filter_works=True)

        # Process common elements
        pairs = list(zip(filtered_df['work_id_one'], filtered_df['work_id_two']))
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
                    'p_value': 0.0,
                    'cited_by_count_one': filtered_df.iloc[i]['cited_by_count_one'],
                    'cited_by_count_two': filtered_df.iloc[i]['cited_by_count_two'],
                    'combined_cited_by_count': filtered_df.iloc[i]['combined_cited_by_count']
                })

        # Calculate total scores
        insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)

        # Convert insert_data back to DataFrame
        filtered_df = pd.DataFrame(insert_data)

        filtered_df['source'] = 'works_common_authors'

        print("\nFinal schema:")
        print(filtered_df.dtypes)
        print("\nFirst 20 rows of final dataframe:")
        print(filtered_df.head(20).to_string())

        final_rows = len(filtered_df)
        print(f"Final number of rows: {final_rows}")
        print(f"Removed {initial_rows - final_rows} rows")

        common_authors_file_filtered = os.path.join(self.datasets_directory, "works_common_authors_filtered.parquet")

        # Save the filtered DataFrame
        filtered_df.to_parquet(common_authors_file_filtered, index=False)
        print(f"Filtered common authors file saved to {common_authors_file}")



    @measure_time
    def restructure_augmented_data(self):
        self.create_augmented_data()

        augmented_data_file = os.path.join(self.datasets_directory, "works_augmented_data.parquet")
        print("Filtering augmented data file...")

        # Read the parquet file
        df = pd.read_parquet(augmented_data_file)

        print("Schema of augmented_data_file:")
        print(df.dtypes)
        print("\nFirst 20 rows of augmented_data_file:")
        print(df.head(20).to_string())

        initial_rows = len(df)
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
        filtered_df = df[df.apply(keep_row, axis=1)]

        # Reset the counter for the final count
        work_id_counter = {}
        for _, row in filtered_df.iterrows():
            work_id_counter[row['work_id_one']] = work_id_counter.get(row['work_id_one'], 0) + 1
            work_id_counter[row['work_id_two']] = work_id_counter.get(row['work_id_two'], 0) + 1

        # Process common elements
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

            return pd.Series({
                'common_uni_grams': common_unigrams,
                'common_bi_grams': common_bigrams,
                'common_field': True,
                'common_subfield': True
            })

        # Apply the processing to each row
        processed_df = filtered_df.apply(process_common_elements, axis=1)

        # Combine the processed data with the original filtered data
        result_df = pd.concat([filtered_df, processed_df], axis=1)

        # Prepare data for insertion
        insert_data = result_df.to_dict('records')

        gc.collect()

        unigrams_df, bigrams_df = self.load_ngrams()

        # Calculate total scores
        insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)

        # Convert insert_data back to DataFrame
        filtered_df = pd.DataFrame(insert_data)

        filtered_df['source'] = 'works_augmented_data'

        print("\nFinal schema:")
        print(filtered_df.dtypes)
        print("\nFirst 20 rows of final dataframe:")
        print(filtered_df.head(20).to_string())

        final_rows = len(filtered_df)
        print(f"Final number of rows: {final_rows}")
        print(f"Removed {initial_rows - final_rows} rows")

        # Print work_id occurrence statistics
        print("\nWork ID occurrence statistics:")
        print(f"Number of unique work_ids: {len(work_id_counter)}")
        print(f"Number of work_ids appearing once: {sum(1 for count in work_id_counter.values() if count == 1)}")
        print(f"Number of work_ids appearing twice: {sum(1 for count in work_id_counter.values() if count == 2)}")

        # Save the filtered DataFrame
        filtered_df.to_parquet(augmented_data_file, index=False)
        print(f"Filtered augmented data file saved to {augmented_data_file}")

        return augmented_data_file

    @measure_time
    def create_augmented_data(self):
        print("Creating augmented data...")
        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        df = pd.read_parquet(works_file)

        # Select top 100% of rows
        top_100_percent = df.head(int(len(df) * 1.0))

        # Define probability map for each augmentation type
        augmentation_prob_map = {
            'full_title': 0.05,
            'full_title_field': 0.15,
            'author_field': 0.15,
            'all_authors_field': 0.10,
            'one_author_field_subfield': 0.10,
            'two_authors_field_subfield': 0.10,
            'two_authors_field': 0.15,
            'full_title_field_subfield': 0.05,
            'all_authors_field_subfield': 0.05,
            'field': 0.002,
            'field_subfield': 0.001,
            'trigram_title': 0.10,
            'trigram_title_field': 0.10,
            'trigram_field_subfield': 0.05,
        }

        augmented_pairs = []

        for _, row in top_100_percent.iterrows():
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

            # Only add the pair if an augmented string was created
            if augmented_string:
                augmented_pairs.append({
                    'work_id_one': row['work_id'],
                    'full_string_one': full_string,
                    'work_id_two': row['work_id'],
                    'full_string_two': augmented_string,
                    'label': 'similar',
                    'label_int': 1,
                    'augmentation_type': selected_type,
                    'p_value': 0.0
                })

        # Create DataFrame from augmented pairs
        augmented_df = pd.DataFrame(augmented_pairs)

        # Save to parquet file
        output_file = os.path.join(self.datasets_directory, 'works_augmented_data.parquet')
        augmented_df.to_parquet(output_file, index=False)

        print(f"Augmented data created and saved to {output_file}")
        print(f"Total augmented pairs: {len(augmented_df)}")

        # Print counts for each augmentation type
        print("\nAugmentation type counts:")
        print(augmented_df['augmentation_type'].value_counts())

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
                           vectorized_subfields):
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
                    'p_value': 0.0
                })
        return insert_data

    @measure_time
    def restructure_augmented_data(self):
        self.create_augmented_data()

        augmented_data_file = os.path.join(self.datasets_directory, "works_augmented_data.parquet")
        print("Filtering augmented data file...")

        # Read the parquet file
        df = pd.read_parquet(augmented_data_file)

        print("Schema of augmented_data_file:")
        print(df.dtypes)
        print("\nFirst 20 rows of augmented_data_file:")
        print(df.head(100).to_string())

        print("\nLast 20 rows of augmented_data_file:")
        print(df.tail(100).to_string())

        initial_rows = len(df)
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
        filtered_df = df[df.apply(keep_row, axis=1)]

        # Reset the counter for the final count
        work_id_counter = {}
        for _, row in filtered_df.iterrows():
            work_id_counter[row['work_id_one']] = work_id_counter.get(row['work_id_one'], 0) + 1
            work_id_counter[row['work_id_two']] = work_id_counter.get(row['work_id_two'], 0) + 1

        # Process common elements
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

            return pd.Series({
                'common_uni_grams': common_unigrams,
                'common_bi_grams': common_bigrams,
                'common_field': True,
                'common_subfield': True
            })

        # Apply the processing to each row
        processed_df = filtered_df.apply(process_common_elements, axis=1)

        # Combine the processed data with the original filtered data
        result_df = pd.concat([filtered_df, processed_df], axis=1)

        # Prepare data for insertion
        insert_data = result_df.to_dict('records')

        gc.collect()

        unigrams_df, bigrams_df = self.load_ngrams()

        # Calculate total scores
        insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)

        # Convert insert_data back to DataFrame
        filtered_df = pd.DataFrame(insert_data)

        filtered_df['source'] = 'works_augmented_data'

        print("\nFinal schema:")
        print(filtered_df.dtypes)
        print("\nFirst 20 rows of final dataframe:")
        print(filtered_df.head(20).to_string())

        final_rows = len(filtered_df)
        print(f"Final number of rows: {final_rows}")
        print(f"Removed {initial_rows - final_rows} rows")

        # Print work_id occurrence statistics
        print("\nWork ID occurrence statistics:")
        print(f"Number of unique work_ids: {len(work_id_counter)}")
        print(f"Number of work_ids appearing once: {sum(1 for count in work_id_counter.values() if count == 1)}")
        print(f"Number of work_ids appearing twice: {sum(1 for count in work_id_counter.values() if count == 2)}")

        # Save the filtered DataFrame
        filtered_df.to_parquet(augmented_data_file, index=False)
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
        ngram_df = pd.DataFrame(ngram_data, columns=[f'{ngram_type}_type', 'count', 'score'])

        file_path = os.path.join(output_dir, f'{ngram_type}_data.parquet')
        ngram_df.to_parquet(file_path, index=False)

        print(f"{ngram_type.capitalize()} data saved to {file_path}. Total rows: {len(ngram_df)}")


    @measure_time
    def preprocess_and_calculate_ngrams(self, remove_single_counts=False, max_rows=500_000_000):
        print("Processing all works and calculating n-grams...")
        output_dir = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\datasets"
        os.makedirs(output_dir, exist_ok=True)

        unigram_counts = Counter()
        bigram_counts = Counter()
        total_processed = 0

        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        df = pd.read_parquet(works_file)

        for _, row in df.iterrows():
            uni_grams = row['uni_grams']
            bi_grams = row['bi_grams']
            unigram_counts.update(uni_grams)
            bigram_counts.update(bi_grams)
            total_processed += 1

            if total_processed >= max_rows:
                break

            if total_processed % 100000 == 0:
                print(f"Processed {total_processed} works")

        print("Counting completed.")

        if remove_single_counts:
            print("Removing items with count = 1...")
            unigram_counts = self.remove_single_count_items(unigram_counts, min_count=1)
            bigram_counts = self.remove_single_count_items(bigram_counts, min_count=1)

        print("Saving data...")

        # Save unigram data
        self.save_ngram_data(unigram_counts, 'unigram', output_dir)

        # Save bigram data
        self.save_ngram_data(bigram_counts, 'bigram', output_dir)

        print(f"Finished processing all works and saving n-gram data. Total works processed: {total_processed}")

    @measure_time
    def remove_single_count_items(self, counter, min_count=1):
        return Counter({item: count for item, count in counter.items() if count > min_count})

    @measure_time
    def save_ngram_data(self, ngram_counts, ngram_type, output_dir):
        ngram_data = [
            (gram, count, 0.0) for gram, count in ngram_counts.items()
        ]
        ngram_df = pd.DataFrame(ngram_data, columns=[f'{ngram_type}_type', 'count', 'score'])

        file_path = os.path.join(output_dir, f'{ngram_type}_data.parquet')
        ngram_df.to_parquet(file_path, index=False)

        print(f"{ngram_type.capitalize()} data saved to {file_path}. Total rows: {len(ngram_df)}")

    @measure_time
    def batch_update_ngram_scores(self):
        for is_bigram in [False, True]:
            file_path = self.bigram_data_file if is_bigram else self.unigram_data_file
            gram_type = 'bigram_type' if is_bigram else 'unigram_type'

            print(f"Processing {gram_type}...")
            print(f"file_path {file_path} ...")

            # Load parquet file
            df = pd.read_parquet(file_path)

            # Find and print duplicates
            self.find_and_print_duplicates(df, gram_type)

            # Calculate new scores
            df = self.calculate_ngram_scores_from_counts(df, is_bigram)

            # Save updated parquet file
            df.to_parquet(file_path, index=False)

            print(f"Finished updating {gram_type} scores. Total rows updated: {len(df)}")

        print("Finished updating n-gram scores.")

    @measure_time
    def calculate_ngram_scores_from_counts(self, df, is_bigram):
        multiplier = 20.0 if is_bigram else 20.0
        df['score'] = np.round(
            multiplier / (np.log((df['count']) + 2) - 1 / np.log(df['count'] + 3) + df['count'] / 100000), 4
        )
        return df

    @measure_time
    def find_and_print_duplicates(self, df, gram_type):
        # Find duplicates
        duplicates = df[df.duplicated(subset=[gram_type], keep=False)]

        if duplicates.empty:
            print(f"No duplicates found in {gram_type}")
        else:
            print(f"Found {len(duplicates)} duplicate entries in {gram_type}:")
            for _, group in duplicates.groupby(gram_type):
                print(group)

        return duplicates

    @measure_time
    def create_sentence_embeddings(self, works_batch_size=100_000):
        works_file = self.works_all_collected_file
        df = pd.read_parquet(works_file)

        total_works = len(df)
        total_batches = (total_works + works_batch_size - 1) // works_batch_size

        model = SentenceTransformer(self.model_path)
        model = DataParallel(model)
        model.to('cuda')

        for batch_num in range(total_batches):
            torch.cuda.empty_cache()
            start_idx = batch_num * works_batch_size
            end_idx = min((batch_num + 1) * works_batch_size, total_works)
            batch_works = df.iloc[start_idx:end_idx]

            sentences = []
            work_ids = []
            work_int_ids = []

            for _, work in batch_works.iterrows():
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

            batch_data = pd.DataFrame({
                'work_id': work_ids,
                'work_int_id': work_int_ids,
                'work_sentence': sentences,
                'work_embedding': embeddings
            })

            file_name = f'work_embeddings_batch_{batch_num}.parquet'
            file_path = os.path.join(self.embeddings_dir, file_name)
            batch_data.to_parquet(file_path, index=False)

            print(f"Processed batch {batch_num + 1}/{total_batches}, saved to {file_path}")

        print(f"Sentence embeddings created and saved in {self.embeddings_dir}")
        print(f"Total works processed: {total_works}")

    def create_sentence_work(self, work_info):
        display_name = work_info.get('title_string', '')
        author_names = work_info.get('authors_string', '').split()
        field = work_info.get('field_string', '')
        subfield = work_info.get('subfield_string', '')

        query_string = f"{display_name} {' '.join(author_names)} {field} {subfield}"
        return query_string



    def sort_files_numerically(self):
        files = os.listdir(self.embeddings_output_directory)
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
            file_path = os.path.join(self.input_directory, file)
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

        index_path = os.path.join(self.input_directory, "works_index.bin")
        faiss.write_index(index, index_path)

        mapping_df = pd.DataFrame({
            'works_int_id': work_int_ids,
            'work_id': work_ids,
        })

        mapping_path = os.path.join(self.input_directory, "works_id_mapping.parquet")
        mapping_df.to_parquet(mapping_path, index=False)

        print(f"FAISS index created and saved to {index_path}")
        print(f"ID mapping saved to {mapping_path}")

        return index_path, mapping_path  # Add this line to return the paths


    def calculate_index_parameters(self, collection_size):
        if collection_size < 1_000_000:
            nlist = int(4 * math.sqrt(collection_size))
            return f"IVF{nlist}", nlist, None
        elif 1_000_000 <= collection_size < 10_000_000:
            return "IVF65536_HNSW32", 65536, 32
        elif 10_000_000 <= collection_size < 25_000_000:
            return "IVF262144_HNSW32", 262144, 32
        else:
            return "IVF1048576_HNSW32", 1048576, 32


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
    def create_faiss_index(self, embeddings, int_ids, item_ids, collection_name):
        d = embeddings.shape[1]
        collection_size = len(int_ids)
        index_type, nlist, hnsw_m = self.calculate_index_parameters(collection_size)

        if "HNSW" in index_type:
            quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, 32, 8)  # Reduced from 32 to 16 sub-quantizers
        else:
            index = faiss.index_factory(d, index_type + ",PQ32")  # Reduced from PQ32 to PQ16

        # Use ScalarQuantizer for very small dimensions
        if d <= 32:
            index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)

        index.train(embeddings)
        index.add_with_ids(embeddings, np.array(int_ids))
        index.nprobe = min(32, nlist // 4)  # Reduced max nprobe from 64 to 32

        return index

    @measure_time
    def add_remaining_vectors_to_index(self, embedding_parquet_directory, output_directory, index_path, mapping_path,
                                       collection_name, batch_size=2000000):
        index = faiss.read_index(index_path)
        mapping_df = pd.read_parquet(mapping_path)

        max_int_id = mapping_df['works_int_id'].max()
        parquet_files = [f for f in os.listdir(embedding_parquet_directory) if
                         f.endswith(".parquet") and '_embeddings' in f]
        remaining_files = [f for f in parquet_files if int(f.split("_")[-1].split(".")[0]) > max_int_id]
        remaining_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        all_data = []
        file_names = []

        for i, file in enumerate(remaining_files):
            print(f"Processing file {i + 1}/{len(remaining_files)}: {file}")
            file_path = os.path.join(embedding_parquet_directory, file)
            table = pq.read_table(file_path)
            data = table.to_pandas()
            all_data.extend(data.to_dict('records'))
            file_names.extend([file] * len(data))

            if len(all_data) >= batch_size:
                self.process_batch(all_data, file_names, index, mapping_df, collection_name)
                all_data = []
                file_names = []
                gc.collect()

        if all_data:
            self.process_batch(all_data, file_names, index, mapping_df, collection_name)

        updated_index_path = os.path.join(output_directory, f"works_index_updated.bin")
        faiss.write_index(index, updated_index_path)

        updated_mapping_path = os.path.join(output_directory, f"works_id_mapping_updated.parquet")
        mapping_df.to_parquet(updated_mapping_path, index=False)

        print(f"Remaining vectors added to the index for {collection_name}.")
        print(f"Updated index saved to {updated_index_path}")
        print(f"Updated mapping saved to {updated_mapping_path}")

    def process_batch(self, all_data, file_names, index, mapping_df, collection_name):
        int_ids = [item['work_int_id'] for item in all_data]
        item_ids = [item['work_id'] for item in all_data]
        embeddings = np.array([item['embedding'] for item in all_data])

        index.add(embeddings)
        additional_df = pd.DataFrame({
            'file_name': file_names,
            'works_int_id': int_ids,
            'work_id': item_ids,
        })
        return pd.concat([mapping_df, additional_df], ignore_index=True)


    @measure_time
    def generate_training_pairs(self, batch_size=512, initial_k=128):
        """
        TODO: Consider generate training pairs:
            We want to do the following:
            We want to give pair scores. These are rough scores that will help us select the right pair later on? IDK.

        :param batch_size:
        :param initial_k:
        :return:
        """


        self.load_index_and_mapping()
        self.load_works_data()
        unigrams_df, bigrams_df = self.load_ngrams()

        works_filtered_df = pd.read_parquet(self.works_all_collected_file)

        pairs_generated = 0
        processed_works = set()

        index_path = self.index_works_file
        index = faiss.read_index(index_path)

        mapping_path = self.id_mapping_works_file
        mapping_df = pd.read_feather(mapping_path)

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
                                       (~self.works_df.index.isin(processed_works))
                                       ].index[:batch_size].tolist()

            if not unprocessed_work_ids:
                print("No more unprocessed works found.")
                break

            similar_works_df = self.batch_search_similar_works(unprocessed_work_ids, k, index, faiss_to_works_id)

            all_pairs = []
            work_pair_count = {}
            print("Length of processed works: ", len(processed_works))
            gc.collect()

            # In the main loop
            for query_work_id in tqdm(unprocessed_work_ids, desc="Processing work IDs"):
                similar_works = similar_works_df[similar_works_df['query_work_id'] == query_work_id][
                    'similar_work_id'].tolist()

                valid_pairs, counts, new_work_pair_count = self.filter_and_count_pairs(similar_works, unigrams_dict,
                                                                                       self.work_details, k)

                all_pairs.extend(valid_pairs)

                # Update the global work_pair_count
                for work_id, count in new_work_pair_count.items():
                    work_pair_count[work_id] = work_pair_count.get(work_id, 0) + count

            if k > 128:
                min_count = 3
                max_appearances = 6
            elif 128 >= k > 64:
                min_count = 2
                max_appearances = 5
            else:
                min_count = 2
                max_appearances = 4

            filtered_pairs = self.filter_pairs_by_count(all_pairs, work_pair_count, cited_by_count_map, min_count=min_count)

            print(f"Total pairs after filtering for min_count req of {min_count}  or more: {len(filtered_pairs)}")

            all_pairs = self.filter_pairs_by_appearance(filtered_pairs, cited_by_count_map, max_appearances=max_appearances)

            print(f"Total pairs after filtering out max_appearances counts over {max_appearances}: {len(all_pairs)}")

            work_ids = set([work_id for pair in all_pairs for work_id in pair])
            work_details = self.fetch_work_details(work_ids, works_filtered_df)

            vectorized_unigrams, vectorized_bigrams, vectorized_fields, vectorized_subfields = self.process_and_vectorize_common_elements(
                work_details, all_pairs)

            insert_data = self.create_insert_data(all_pairs, work_details, vectorized_unigrams, vectorized_bigrams,
                                                  vectorized_fields, vectorized_subfields)
            insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)
            insert_data = self.process_p_values(insert_data)

            # insert_data['source'] = 'works_knn_search'

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
            elif k < 100 and (common_unigrams_count >= 1 or common_field or rand_num > 0.95):
                is_valid = True
            elif k < 50 and (common_unigrams_count >= 1 or common_field or rand_num > 0.90):
                is_valid = True
            elif common_unigrams_count >= 1 and rand_num > 0.8:
                counts["common_1"] += 1
                is_valid = True
            elif common_field and rand_num > 0.8:
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
    def filter_pairs_by_count(self, all_pairs, work_pair_count, cited_by_count_map, min_count=4):
        # Filter pairs based on minimum count
        filtered_pairs = [pair for pair in all_pairs if
                          work_pair_count.get(pair[0], 0) >= min_count and work_pair_count.get(pair[1], 0) >= min_count]

        # Sort pairs by combined cited_by_count in descending order
        sorted_pairs = sorted(filtered_pairs,
                              key=lambda x: (cited_by_count_map.get(x[0], 0) + cited_by_count_map.get(x[1], 0)),
                              reverse=True)

        return sorted_pairs

    @measure_time
    def filter_pairs_by_appearance(self, filtered_pairs, cited_by_count_map, max_appearances=6):
        final_pairs = []
        work_appearance_count = {}

        # Sort pairs by combined cited_by_count in descending order
        sorted_pairs = sorted(filtered_pairs,
                              key=lambda x: (cited_by_count_map.get(x[0], 0) + cited_by_count_map.get(x[1], 0)),
                              reverse=True)

        for i, pair in enumerate(sorted_pairs):
            if work_appearance_count.get(pair[0], 0) < max_appearances and \
                    work_appearance_count.get(pair[1], 0) < max_appearances:
                final_pairs.append(pair)
                work_appearance_count[pair[0]] = work_appearance_count.get(pair[0], 0) + 1
                work_appearance_count[pair[1]] = work_appearance_count.get(pair[1], 0) + 1

        return final_pairs

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
        self.works_df.loc[self.works_df.index.isin(all_work_ids), 'work_id_search_count'] += 1

        print(f"Updated work_id_search_count for {len(all_work_ids)} works")

    @measure_time
    def calculate_total_scores(self, insert_data, unigrams_df, bigrams_df):
        """
        TODO: This method needs to be changed. We will be multiplying by the total_score by scalar_multiple,
            so the more unigrams in common, the bigger the field/subfield multiplier we have.
            This method will need careful treatment.

        TODO: Instead of using avg_gram_score, we wanna use the sum of the gram scores

        :param insert_data:
        :param unigrams_df:
        :param bigrams_df:
        :return:
        """

        df = pd.DataFrame(insert_data)

        df['unigram_score'] = self.vectorized_gram_scores(df['common_uni_grams'], unigrams_df)
        df['bigram_score'] = self.vectorized_gram_scores(df['common_bi_grams'], bigrams_df)

        # Calculate average gram score
        df['avg_gram_score'] = (df['unigram_score'] + df['bigram_score']) / 2

        scalar_multiplier = 0.05
        df['field_score'] = df['common_field'] * (3.0 + 2.0 * scalar_multiplier * df['avg_gram_score'])
        df['subfield_score'] = df['common_subfield'] * (1.0 + scalar_multiplier * df['avg_gram_score'])
        df['total_score'] = df['unigram_score'] + df['bigram_score'] + df['field_score'] + df['subfield_score']

        # Convert back to list of dictionaries
        return df.to_dict('records')

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
        if not isinstance(insert_data, pd.DataFrame):
            df = pd.DataFrame(insert_data)
        else:
            df = insert_data.copy()

        # Vectorized calculation of z_scores
        df['z_score'] = (df['total_score'] - mean_score) / std_score

        # Vectorized calculation of p_values
        df['p_value'] = 1 - norm.cdf(df['z_score'])

        # If the original input was a list of dictionaries, convert back
        if not isinstance(insert_data, pd.DataFrame):
            return df.to_dict('records')
        else:
            return df

    @measure_time
    def filter_by_p_value(self, insert_data):
        if isinstance(insert_data, pd.DataFrame):
            return insert_data[(insert_data['p_value'] <= 0.49) | (insert_data['p_value'] >= 0.51)]
        elif isinstance(insert_data, list):
            return [item for item in insert_data if item['p_value'] <= 0.49 or item['p_value'] >= 0.51]
        else:
            raise TypeError("insert_data must be a DataFrame or a list of dictionaries")

    @measure_time
    def create_work_id_count(self, filtered_data):
        if isinstance(filtered_data, pd.DataFrame):
            df = filtered_data
        elif isinstance(filtered_data, list):
            df = pd.DataFrame(filtered_data)
        else:
            raise TypeError("filtered_data must be a DataFrame or a list of dictionaries")

        work_id_count = {}
        for _, row in df.iterrows():
            for work_id in [row['work_id_one'], row['work_id_two']]:
                if work_id not in work_id_count:
                    work_id_count[work_id] = {'above': 0, 'below': 0}
                if row['p_value'] > 0.5:
                    work_id_count[work_id]['above'] += 1
                else:
                    work_id_count[work_id]['below'] += 1

        return work_id_count

    @measure_time
    def remove_single_occurrence_pairs(self, filtered_data, work_id_count):
        if isinstance(filtered_data, pd.DataFrame):
            df = filtered_data
        elif isinstance(filtered_data, list):
            df = pd.DataFrame(filtered_data)
        else:
            raise TypeError("filtered_data must be a DataFrame or a list of dictionaries")

        def has_both_occurrences(work_id):
            return work_id_count[work_id]['above'] > 0 and work_id_count[work_id]['below'] > 0

        filtered_df = df[df['work_id_one'].apply(has_both_occurrences) &
                         df['work_id_two'].apply(has_both_occurrences)]

        if isinstance(filtered_data, list):
            return filtered_df.to_dict('records')
        else:
            return filtered_df


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
        self.faiss_to_work_id_mapping = pd.read_feather(mapping_path)

    @measure_time
    def load_works_data(self, duplicates_check=True):
        self.works_df = pd.read_parquet(self.works_all_collected_file)

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
            self.works_df.reset_index(drop=True).to_parquet(works_file, index=False)
        except Exception as e:
            print("error: ", e)
            try:
                self.works_df.reset_index().to_parquet(works_file, index=False)
            except Exception as e:
                print("error: ", e)
                try:
                    self.works_df.to_parquet(works_file, index=False)
                except Exception as e:
                    print("error: ", e)

        print(f"Updated work_id_search_count for {len(self.work_id_search_count)} works in Parquet file")
        siamese_file = os.path.join(self.datasets_directory, "works_knn_search.parquet")
        hard_negatives_file = os.path.join(self.datasets_directory, "hard_negatives_pool.parquet")

        # Save siamese data
        if not os.path.exists(siamese_file):
            columns = [
                'work_id_one', 'full_string_one', 'work_id_two', 'full_string_two',
                'common_uni_grams', 'common_bi_grams', 'common_field', 'common_subfield',
                'total_score', 'label', 'label_int', 'p_value'
            ]
            df = pd.DataFrame(self.works_knn_search, columns=columns)
        else:
            existing_df = pd.read_parquet(siamese_file)
            new_df = pd.DataFrame(self.works_knn_search)
            df = pd.concat([existing_df, new_df], ignore_index=True)

        df.to_parquet(siamese_file, index=False)
        print(f"Saved {len(self.works_knn_search)} entries to works_knn_search data Parquet file")

        # Save hard negatives pool
        if os.path.exists(hard_negatives_file):
            hard_negatives_df = pd.read_parquet(hard_negatives_file)
            print(f"Saved {len(hard_negatives_df)} entries to hard negatives pool Parquet file")
        else:
            print("No hard negatives pool file found")

        # Clear the in-memory data
        self.work_id_search_count.clear()
        self.works_knn_search.clear()

        # Force garbage collection
        gc.collect()


    def load_ngrams(self):

        unigrams_df = pd.read_feather(self.unigram_data_file)
        bigrams_df = pd.read_feather(self.bigram_data_file)

        return unigrams_df, bigrams_df

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
            common_fields.append(work1.get('field_string') == work2.get('field_string'))
            common_subfields.append(work1.get('subfield_string') == work2.get('subfield_string'))

        return common_unigrams, common_bigrams, common_fields, common_subfields

    @measure_time
    def vectorized_common_unigrams(self, common_unigrams):
        return [list(unigrams) for unigrams in common_unigrams]

    @measure_time
    def vectorized_common_bigrams(self, common_bigrams):
        return [list(bigrams) for bigrams in common_bigrams]

    @measure_time
    def vectorized_common_fields(self, common_fields):
        return np.array(common_fields, dtype=int)

    @measure_time
    def vectorized_common_subfields(self, common_subfields):
        return np.array(common_subfields, dtype=int)

    @measure_time
    def batch_search_similar_works(self, work_ids, k, index, faiss_to_works_id):

        work_embeddings = self.batch_encode_works(
            [self.create_sentence_work(self.work_details[work_id]) for work_id in work_ids])

        distances, indices = self.perform_batch_search(index, work_embeddings, k)

        results = []
        for i, work_id in enumerate(work_ids):
            for j in range(k):
                faiss_idx = int(indices[i][j])
                try:
                    # Use the lookup dictionary instead of DataFrame loc
                    similar_work_id = faiss_to_works_id[faiss_idx]
                    results.append({
                        'query_work_id': work_id,
                        'similar_work_id': similar_work_id,
                        'distance': float(distances[i][j])
                    })
                except KeyError:
                    print(f"Warning: No mapping found for FAISS index {faiss_idx}")

        return pd.DataFrame(results)

    @measure_time
    def fetch_work_details(self, work_ids, works_filtered_df, truncated=False, filter_works=True):
        result = {}

        if filter_works:
            # Filter the DataFrame to include only the specified work_ids
            df_to_process = works_filtered_df[works_filtered_df['work_id'].isin(work_ids)]
        else:
            # Use the entire DataFrame without filtering
            df_to_process = works_filtered_df

        for _, row in df_to_process.iterrows():
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
    def vectorized_gram_scores(self, gram_series, gram_df):
        all_grams = set([gram for gram_set in gram_series for gram in gram_set])
        scores_dict = self.get_gram_scores(all_grams, gram_df)
        return gram_series.apply(lambda gram_set: sum(scores_dict.get(gram, 0.01) for gram in gram_set))

    @measure_time
    def get_gram_scores(self, grams, gram_df):
        gram_type = "unigram_type" if 'unigram_type' in gram_df.columns else "bigram_type"
        scores = gram_df[gram_df[gram_type].isin(grams)].set_index(gram_type)['score'].to_dict()
        return {gram: float(scores.get(gram, 2.5)) for gram in grams}

    @measure_time
    def batch_encode_works(self, work_strings, batch_size=8):
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
        works_df = pd.read_parquet(self.works_all_collected_file)

        cited_by_count_map = dict(zip(works_df['work_id'], works_df['cited_by_count']))
        work_id_to_title = dict(zip(works_df['work_id'], works_df['title_string']))

        # Get the set of stop words
        stop_words = self.get_stop_words()

        # Initialize list to store common title pairs
        self.common_title_pairs = []

        self.process_file_for_common_titles(self.works_common_authors_file, work_id_to_title, stop_words)

        self.process_file_for_common_titles(self.works_augmented_data_file, work_id_to_title, stop_words)

        self.process_file_for_common_titles(self.works_knn_search_file, work_id_to_title, stop_words)

        # Create a DataFrame from the common title pairs
        common_title_df = pd.DataFrame(self.common_title_pairs)

        # Save the DataFrame as a parquet file

        common_title_df.to_parquet(self.works_common_titles_file, index=False)

        print(f"Created common_title_works.parquet with {len(common_title_df)} pairs")

        self.common_title_pairs = []

        gc.collect()

    def process_file_for_common_titles(self, file_path, work_id_to_title, stop_words):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        df = pd.read_parquet(file_path)

        for _, row in tqdm(df.iterrows(), desc=f"Processing {os.path.basename(file_path)}"):
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
                continue

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
            if (len(common_title_unigrams) >= unigram_threshold or
                    len(common_title_bigrams) >= 2 or
                    len(common_title_trigrams) >= 1):
                self.common_title_pairs.append({
                    'work_id_one': work_id_one,
                    'work_id_two': work_id_two,
                    'common_title_unigrams': list(common_title_unigrams),
                    'common_title_bigrams': list(common_title_bigrams),
                    'common_title_trigrams': list(common_title_trigrams),
                    'total_score': row['total_score'],  # Use the pre-calculated total_score
                    'source': "common_title_works",
                })

        # Force garbage collection
        gc.collect()


    @measure_time
    def generate_all_work_id_pairs_dataset(self, sort_by_distance=True, last_work_int_id=0):
        print("Generating all work ID pairs dataset...")

        # TODO: This lambda line is problematic as it is writing up the full string for augmentation type.
        #  We cannot replace the augmented data tih the full string.

        files = [
            self.works_common_authors_file,
            self.works_common_titles_file,
            self.works_augmented_data_file,
            self.works_knn_search_file
        ]

        all_pairs = []
        work_id_counts = {}

        # First pass: Load all pairs and calculate median score
        for file in files:
            if os.path.exists(file):
                df = pd.read_parquet(file)
                df['source'] = os.path.basename(file).replace('.parquet', '')
                df['augmentation_type'] = df.get('augmentation_type', None)

                # Select only the required columns
                selected_df = df[['work_id_one', 'work_id_two', 'total_score', 'source', 'augmentation_type']]
                all_pairs.extend(selected_df.values.tolist())

            gc.collect()

        # Shuffle all_pairs to mix up the sources
        random.shuffle(all_pairs)

        files_two = [self.softer_negatives_pool_file]

        # First pass: Load all pairs and calculate median score
        for file in files_two:
            if os.path.exists(file):
                df = pd.read_parquet(file)
                df['source'] = os.path.basename(file).replace('.parquet', '')
                df['augmentation_type'] = df.get('augmentation_type', None)

                # Select only the required columns
                selected_df = df[['work_id_one', 'work_id_two', 'total_score', 'source', 'augmentation_type']]
                all_pairs.extend(selected_df.values.tolist())

            gc.collect()

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
        # Convert to DataFrame and continue with the rest of the method...
        columns = ['work_id_one', 'work_id_two', 'total_score', 'source', 'augmentation_type']
        filtered_pairs_df = pd.DataFrame(filtered_pairs, columns=columns)

        # Calculate z-scores and normalize
        mean_score = filtered_pairs_df['total_score'].mean()
        std_score = filtered_pairs_df['total_score'].std()
        filtered_pairs_df['z_score'] = (filtered_pairs_df['total_score'] - mean_score) / std_score
        filtered_pairs_df['normalized_z_score'] = self.sigmoid_normalize(filtered_pairs_df['z_score'])

        file_name = f"triplet_work_ids_only_{last_work_int_id}.parquet"
        output_file = os.path.join(self.datasets_directory, file_name)
        filtered_pairs_df.to_parquet(output_file, index=False)
        print(f"Saved triplet_work_ids_only to {output_file}")

        # Generate triplets
        positive_pairs = filtered_pairs_df[filtered_pairs_df['normalized_z_score'] >= 0]
        negative_pairs = filtered_pairs_df[filtered_pairs_df['normalized_z_score'] < 0]

        triplets = pd.merge(positive_pairs, negative_pairs, on='work_id_one', suffixes=('_pos', '_neg'))
        triplets = triplets[triplets['work_id_two_pos'] != triplets['work_id_two_neg']]
        triplets = triplets.drop(columns=['z_score_pos', 'z_score_neg'])

        # Calculate max_pos_neg_distance
        triplets['max_pos_neg_distance'] = triplets['normalized_z_score_pos'] - triplets['normalized_z_score_neg']

        # Sort by max_pos_neg_distance if requested
        if sort_by_distance:
            triplets = triplets.sort_values('max_pos_neg_distance', ascending=False)

        # Rename columns for final output
        triplets = triplets.rename(columns={
            'work_id_one': 'anchor',
            'work_id_two_pos': 'positive',
            'work_id_two_neg': 'negative',
            'normalized_z_score_pos': 'z_score_pos',
            'normalized_z_score_neg': 'z_score_neg',
            'total_score_pos': 'total_score_pos',
            'total_score_neg': 'total_score_neg',
            'source_pos': 'source_pos',
            'source_neg': 'source_neg',
            'augmentation_type_pos': 'augmentation_type_pos',
            'augmentation_type_neg': 'augmentation_type_neg'
        })

        # Fetch work details and create full strings
        works_filtered_df = pd.read_parquet(self.works_all_collected_file)

        all_work_ids = set(triplets['anchor']) | set(triplets['positive']) | set(triplets['negative'])
        work_details = self.fetch_work_details(all_work_ids, works_filtered_df, truncated=True)

        # Create full strings
        triplets['anchor_string'] = triplets['anchor'].map(lambda x: self.create_full_string(work_details.get(x, {})))
        triplets['positive_string'] = triplets['positive'].map(
            lambda x: self.create_full_string(work_details.get(x, {})))
        triplets['negative_string'] = triplets['negative'].map(
            lambda x: self.create_full_string(work_details.get(x, {})))

        # augmented_df = pd.read_parquet(self.works_augmented_data_file)

        # Quality control check: Remove duplicate columns
        triplets = triplets.loc[:, ~triplets.columns.duplicated()]


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

        # Ensure all columns exist and select only the final columns
        triplets = triplets[final_columns]

        # Initialize dictionaries to keep track of occurrences
        anchor_occurrences = {}
        positive_occurrences = {}
        negative_occurrences = {}

        # List to store the filtered triplets
        filtered_triplets = []

        for _, row in triplets.iterrows():
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
        triplets = pd.DataFrame(filtered_triplets)

        # Print head and tail of the DataFrame
        print("\nHead of the DataFrame (20 rows):")
        print(triplets.head(20).to_string())
        print("\nTail of the DataFrame (20 rows):")
        print(triplets.tail(20).to_string())
        print("length of triplets: ", len(triplets))

        print("Original number of triplets:", len(triplets))
        print("Number of triplets after filtering:", len(filtered_triplets))

        # Print column names after final selection
        print("\nFinal column names:")
        print(triplets.columns.tolist())

        triplets.to_parquet(self.triplets_file, index=False)
        print(f"\nSaved triplets to {self.triplets_file}")

    def sigmoid_normalize(self, x):
        return (2 / (1 + np.exp(-x))) - 1



    @measure_time
    def triplets_quality_control_statistics(self):
        print("Performing quality control statistics on triplets...")

        # Load the triplets parquet file
        triplets_file = os.path.join(self.datasets_directory, "triplets.parquet")
        if not os.path.exists(triplets_file):
            print(f"Error: Triplets file not found at {triplets_file}")
            return

        triplets_df = pd.read_parquet(triplets_file)

        # Count occurrences of each work_id
        all_work_ids = pd.concat([
            triplets_df['anchor'],
            triplets_df['positive'],
            triplets_df['negative']
        ])
        work_id_counts = all_work_ids.value_counts()

        # Count work_ids appearing twice in the same row
        same_row_duplicates = (
                (triplets_df['anchor'] == triplets_df['positive']) |
                (triplets_df['anchor'] == triplets_df['negative']) |
                (triplets_df['positive'] == triplets_df['negative'])).sum()

        # Count work_ids appearing different number of times
        appear_once = (work_id_counts == 1).sum()
        appear_twice = (work_id_counts == 2).sum()
        appear_thrice = (work_id_counts == 3).sum()
        appear_four_or_more = (work_id_counts >= 4).sum()

        # Print statistics
        print(f"\nTotal number of triplets: {len(triplets_df)}")
        print(f"Total unique work_ids: {len(work_id_counts)}")
        print(f"\nWork_ids appearing twice in the same row: {same_row_duplicates}")
        print(f"Work_ids appearing only once in the dataset: {appear_once}")
        print(f"Work_ids appearing twice in the dataset: {appear_twice}")
        print(f"Work_ids appearing three times in the dataset: {appear_thrice}")
        print(f"Work_ids appearing four or more times in the dataset: {appear_four_or_more}")

        # Additional statistics
        print(f"\nMaximum appearances of a single work_id: {work_id_counts.max()}")
        print("\nDistribution of work_id appearances:")
        appearance_distribution = work_id_counts.value_counts().sort_index()
        for appearances, count in appearance_distribution.items():
            print(f"  {appearances} time(s): {count} work_ids")

        # Check for any missing values
        missing_values = triplets_df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nWarning: Missing values detected:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values detected.")

        # Check for duplicate triplets
        duplicate_triplets = triplets_df.duplicated().sum()
        print(f"\nNumber of duplicate triplets: {duplicate_triplets}")

        # Summary of total_score and z_score distributions
        print("\nSummary statistics for scores:")
        print(triplets_df[['total_score_pos', 'total_score_neg', 'z_score_pos', 'z_score_neg']].describe())

        print("\nQuality control statistics completed.")


if __name__ == "__main__":


    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_file = r"C:\Users\doren\PycharmProjects\CITATION_GRABBER_V2\SENTENCE_ENCODER\works_subfield_checkpoint.json"

    model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\best_model"
    output_directory = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models"
    datasets_directory = f"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\datasets"
    os.makedirs(datasets_directory, exist_ok=True)

    print(datasets_directory)

    run_params = {
        'load_and_print_data': False,
        'collect_all_works_metadata': True,
        'restructure_common_authors':  True,
        'restructure_augmented_data':  True,
        'preprocess_and_calculate_ngrams': False,
        'batch_update_ngram_scores': False,
        'create_sentence_embeddings':  True,
        'calculate_density_scores': False,
        'build_vector_index':  True,
        'generate_training_pairs': True,
        'create_common_title_works': True,
        'generate_all_work_id_pairs_dataset': True,
    }

    encoder = DatasetConstructionSentenceEncoder(
        model_path=model_path,
        output_directory=output_directory,
        datasets_directory=datasets_directory,
        run_params=run_params,
        num_knn_pairs=500_000_000,
        num_works_collected=500_000_000,
        mongo_url="mongodb://localhost:27017/",
        mongo_database_name="OpenAlex",
        mongo_works_collection_name="Works"
    )

    encoder.run()
    encoder.triplets_quality_control_statistics()




