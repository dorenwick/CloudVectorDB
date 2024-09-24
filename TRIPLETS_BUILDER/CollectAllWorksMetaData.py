import os
import random
import gc
import json
from tqdm import tqdm
import polars as pl
import numpy as np
from pymongo import MongoClient

class CollectAllWorksMetadata:
    def __init__(self, datasets_directory, ngrams_directory, mongo_url, mongo_database_name, mongo_works_collection_name, num_works_collected):
        self.datasets_directory = datasets_directory
        self.ngrams_directory = ngrams_directory
        self.mongo_url = mongo_url
        self.mongo_database_name = mongo_database_name
        self.mongo_works_collection_name = mongo_works_collection_name
        self.num_works_collected = num_works_collected
        self.mongo_client = None
        self.mongo_db = None
        self.mongodb_works_collection = None

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

    def collect_all_works_metadata(self, abstract_include=False):
        self.establish_mongodb_connection()
        print("Collecting metadata for all works...")

        total_processed = 0
        batch_size = 10_000
        batch_count = 0
        new_rows = []
        batch_files = []

        projection = {
            "works_int_id": 1,
            "id": 1,
            "display_name": 1,
            "primary_topic": 1,
            "cited_by_count": 1,
            "authorships": 1,
            "abstract_inverted_index": 1,
            "_id": 0
        }

        cursor = self.mongodb_works_collection.find(
            projection=projection
        ).sort("works_int_id", 1).batch_size(batch_size)

        for work in tqdm(cursor, desc="Processing works"):
            work_int_id = work.get('works_int_id')
            work_id = work.get('id')
            title = work.get('display_name', '')
            primary_topic = work.get('primary_topic', {})
            if primary_topic:
                topic = primary_topic.get('topic', {}).get('display_name', '')
                subfield = primary_topic.get('subfield', {}).get('display_name', '')
                field = primary_topic.get('field', {}).get('display_name', '')
            else:
                topic = ''
                subfield = ''
                field = ''

            cited_by_count = work.get('cited_by_count', 0)

            author_names = []
            author_ids = []
            for authorship in work.get('authorships', []):
                author = authorship.get('author', {})
                if 'display_name' in author and 'id' in author:
                    author_names.append(author['display_name'])
                    author_ids.append(author['id'])

            authors_string = ' '.join(author_names)
            text_for_grams = f"{title} {authors_string}"

            if len(text_for_grams) < 8:
                continue

            unigrams = text_for_grams.lower().split()
            bigrams = [f"{unigrams[i]} {unigrams[i + 1]}" for i in range(len(unigrams) - 1)]

            if len(unigrams) < 3:
                continue

            abstract_inverted_index = work.get('abstract_inverted_index', {})
            abstract_string = self.reconstruct_abstract(abstract_inverted_index) if abstract_inverted_index else ''

            new_rows.append({
                'work_id': work_id,
                'work_int_id': work_int_id,
                'title_string': title,
                'authors_string': authors_string,
                'author_names': author_names,
                'field_string': field,
                'subfield_string': subfield,
                'topic_string': topic,
                'abstract_string': abstract_string,
                'unigrams': unigrams,
                'bigrams': bigrams,
                'cited_by_count': cited_by_count,
                'contains_title': bool(title),
                'contains_topic': bool(primary_topic),
                'contains_authors': bool(author_names),
                'contains_abstract': bool(abstract_inverted_index),
                'title_author_length': len(text_for_grams),
            })

            total_processed += 1

            if len(new_rows) >= batch_size:
                batch_file = self.save_batch_to_parquet(new_rows, batch_count)
                batch_files.append(batch_file)
                new_rows = []
                batch_count += 1

            if total_processed % 10_000 == 0:
                print(f"Processed {total_processed} works")
                gc.collect()

            if total_processed >= self.num_works_collected:
                break

        if new_rows:
            batch_file = self.save_batch_to_parquet(new_rows, batch_count)
            batch_files.append(batch_file)

        self.close_mongodb_connection()

        output_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        final_df = self.concatenate_parquet_files(batch_files)
        final_df.write_parquet(output_file)

        print(f"Saved final concatenated Polars DataFrame to {output_file}")

        # Save additional DataFrame with just work_id and work_int_id
        id_mapping_df = final_df.select(['work_id', 'work_int_id'])
        id_mapping_file = os.path.join(self.datasets_directory, "work_id_mapping.parquet")
        id_mapping_df.write_parquet(id_mapping_file)
        print(f"Saved work ID mapping to {id_mapping_file}")

        self.partition_embeddings_by_domain()
        self.concatenate_domain_embeddings()

        return output_file

    def save_batch_to_parquet(self, rows, batch_number):
        df = pl.DataFrame(rows)
        batch_file = os.path.join(self.datasets_directory, f"works_batch_{batch_number}.parquet")
        df.write_parquet(batch_file)
        print(f"Saved batch {batch_number} to {batch_file}")
        return batch_file

    def concatenate_parquet_files(self, file_list):
        dfs = [pl.read_parquet(file) for file in file_list]
        concatenated_df = pl.concat(dfs)
        return concatenated_df

    def reconstruct_abstract(self, abstract_inverted_index):
        if not abstract_inverted_index:
            return ""
        max_position = max(max(positions) for positions in abstract_inverted_index.values())
        words = [''] * (max_position + 1)
        for word, positions in abstract_inverted_index.items():
            for position in positions:
                words[position] = word
        return ' '.join(words).strip()

    def partition_embeddings_by_domain(self):
        print("Partitioning embeddings by domain...")

        with open(os.path.join(self.datasets_directory, 'domain_to_field.json'), 'r') as f:
            domain_to_field = json.load(f)
        field_to_domain = {field: domain for domain, fields in domain_to_field.items() for field in fields}

        domain_dirs = {domain: os.path.join(self.datasets_directory, domain.lower().replace(' ', '_'))
                       for domain in domain_to_field.keys()}
        domain_dirs['No Domain'] = os.path.join(self.datasets_directory, 'no_domain')

        for directory in domain_dirs.values():
            os.makedirs(directory, exist_ok=True)

        batch_files = sorted([f for f in os.listdir(self.datasets_directory) if
                              f.startswith('works_batch_') and f.endswith('.parquet')])

        for batch_file in tqdm(batch_files, desc="Processing batch files"):
            file_path = os.path.join(self.datasets_directory, batch_file)
            df = pl.read_parquet(file_path)

            for domain, directory in domain_dirs.items():
                if domain == 'No Domain':
                    domain_df = df.filter(~pl.col('field_string').is_in(field_to_domain.keys()))
                else:
                    domain_fields = domain_to_field[domain]
                    domain_df = df.filter(pl.col('field_string').is_in(domain_fields))

                if not domain_df.is_empty():
                    output_file = os.path.join(directory, f"domain_embeddings_batch_{batch_file.split('_')[-1]}")
                    domain_df.write_parquet(output_file)

        print("Partitioning complete.")

    def concatenate_domain_embeddings(self):
        print("Concatenating domain embeddings...")

        domain_dirs = [d for d in os.listdir(self.datasets_directory) if
                       os.path.isdir(os.path.join(self.datasets_directory, d))]

        for domain_dir in domain_dirs:
            print(f"Processing {domain_dir}...")
            domain_path = os.path.join(self.datasets_directory, domain_dir)
            parquet_files = [f for f in os.listdir(domain_path) if f.endswith('.parquet')]

            if not parquet_files:
                print(f"No parquet files found in {domain_dir}. Skipping.")
                continue

            dfs = []
            for file in tqdm(parquet_files, desc=f"Reading files in {domain_dir}"):
                file_path = os.path.join(domain_path, file)
                df = pl.read_parquet(file_path)
                dfs.append(df)

            concatenated_df = pl.concat(dfs)
            output_file = os.path.join(self.datasets_directory, f"{domain_dir}_embeddings.parquet")
            concatenated_df.write_parquet(output_file)
            print(f"Saved concatenated embeddings for {domain_dir} to {output_file}")

            # Remove individual batch files to save space
            for file in parquet_files:
                os.remove(os.path.join(domain_path, file))

        print("Concatenation complete.")