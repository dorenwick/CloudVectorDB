import os
import pandas as pd
import psutil
from pymongo import MongoClient
from tqdm import tqdm
import polars as pl

class AuthorCollector:
    """
    I will want a method that concatenates these parquet files into a dataframe, using polars, that has the same columns and schema and all.
    we will save it as "works_common_authors_all.parquet." to the directory: E:\HugeDatasetBackup\cloud_datasets
    and we will remove duplicates from all the files whenever we read them, using list(set(...)) to get rid of duplicates.
    we do that first on the works_common_authors.parquet file (which we load first), and then remove duplicates, and then
    we concatenate the rest, also removing duplicates each time. Then we save the final concatenation of all of these as instructed:

    save it as "works_common_authors_all.parquet." to the directory: E:\HugeDatasetBackup\cloud_datasets

    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_153999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_163999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_173999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_183999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_193999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_203999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_213999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_223999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_233999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_243999999.parquet"
    "E:\HugeDatasetBackup\cloud_datasets\works_common_authors_253999999.parquet"
    E:\HugeDatasetBackup\cloud_datasets\works_common_authors_final.parquet

    """


    def __init__(self,
                 mongodb_works_collection,
                 datasets_directory,
                 mongo_url="mongodb://localhost:27017/",
                 mongo_database_name="OpenAlex",
                 mongo_works_collection_name="Works"):


        self.mongodb_works_collection = mongodb_works_collection
        self.datasets_directory = datasets_directory

        # MongoDB connection
        self.mongo_url = mongo_url
        self.mongo_database_name = mongo_database_name
        self.mongo_works_collection_name = mongo_works_collection_name
        self.mongo_client = None
        self.mongo_db = None



    def collect_common_authors(self, start_from_id=234_000_000):
        self.establish_mongodb_connection()
        print(f"Collecting common authors starting from work_int_id {start_from_id}...")

        author_work_map = {}
        common_author_pairs = []
        total_processed = 0
        batch_size = 100_000

        projection = {
            "id": 1,
            "works_int_id": 1,
            "authorships": 1,
            "_id": 0
        }

        # Query for works with work_int_id >= start_from_id
        query = {"works_int_id": {"$gte": start_from_id}}
        cursor = self.mongodb_works_collection.find(
            query,
            projection=projection
        ).sort("works_int_id", 1).batch_size(batch_size)

        for work in tqdm(cursor, desc="Processing works"):
            work_id = work.get('id')
            work_int_id = work.get('works_int_id')

            for authorship in work.get('authorships', []):
                author = authorship.get('author', {})
                if 'id' in author:
                    author_id = author['id']
                    if author_id in author_work_map:
                        common_author_pairs.append((author_work_map[author_id], work_id))
                    author_work_map[author_id] = work_id

            total_processed += 1

            if total_processed % 100_000 == 0:
                print(f"Processed {total_processed} works. Current work_int_id: {work_int_id}")
                self.print_memory_usage(f"batch {total_processed}")
                print(f"len author work map {len(author_work_map)}")
                print(f"len common author pairs {len(common_author_pairs)}")

            # Remove duplicates every 1 million processed works
            if total_processed % 1_000_000 == 0:
                common_author_pairs = list(set(common_author_pairs))
                print(f"len common author pairs after duplicate removal {len(common_author_pairs)}")

            # Save intermediate results every 10 million processed works
            if total_processed % 10_000_000 == 0:
                self.save_common_authors(common_author_pairs, f"works_common_authors_{work_int_id}.parquet")
                common_author_pairs = []  # Clear the list after saving

        # Save final results
        self.save_common_authors(common_author_pairs, "works_common_authors_final.parquet")

        self.close_mongodb_connection()
        print(f"Finished processing. Total works processed: {total_processed}")

    def save_common_authors(self, common_author_pairs, filename):
        common_authors_file = os.path.join(self.datasets_directory, filename)
        common_authors_df = pl.DataFrame(common_author_pairs, schema=['work_id_one', 'work_id_two'])
        common_authors_df.write_parquet(common_authors_file)
        print(f"Saved {common_authors_df.shape[0]} common author pairs to {common_authors_file}")
        del common_authors_df  # Free up memory

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



    def concatenate_common_authors_files(self):
        base_dir = r"E:\HugeDatasetBackup\cloud_datasets"
        output_file = os.path.join(base_dir, "works_common_authors_all.parquet")

        files_to_concatenate = [
            "works_common_authors.parquet",
            "works_common_authors_153999999.parquet",
            "works_common_authors_163999999.parquet",
            "works_common_authors_173999999.parquet",
            "works_common_authors_183999999.parquet",
            "works_common_authors_193999999.parquet",
            "works_common_authors_203999999.parquet",
            "works_common_authors_213999999.parquet",
            "works_common_authors_223999999.parquet",
            "works_common_authors_233999999.parquet",
            "works_common_authors_243999999.parquet",
            "works_common_authors_253999999.parquet",
            "works_common_authors_final.parquet"
        ]

        combined_df = None

        for file in tqdm(files_to_concatenate, desc="Concatenating files"):
            file_path = os.path.join(base_dir, file)
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)

                # Remove duplicates
                df = df.drop_duplicates()

                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

                # Remove duplicates after concatenation
                combined_df = combined_df.drop_duplicates()

                print(f"Processed {file}. Current total rows: {combined_df.shape[0]}")
                self.print_memory_usage(f"After processing {file}")
            else:
                print(f"File not found: {file_path}")

        if combined_df is not None:
            combined_df.to_parquet(output_file, index=False)
            print(f"Saved concatenated file to {output_file}")
            print(f"Total rows in final file: {combined_df.shape[0]}")
        else:
            print("No files were processed. Check if the file paths are correct.")

    def print_memory_usage(self, location):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage at {location}: {memory_info.rss / 1024 / 1024:.2f} MB")

# Usage
if __name__ == "__main__":
    # Initialize your MongoDB connection and datasets directory
    mongodb_works_collection = None  # Replace with your MongoDB collection
    datasets_directory = r"E:\HugeDatasetBackup\cloud_datasets"

    collector = AuthorCollector(mongodb_works_collection, datasets_directory)
    collector.concatenate_common_authors_files()

