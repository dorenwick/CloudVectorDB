import glob
import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from SENTENCE_ENCODER.DatasetConstructionSentenceEncoder import DatasetConstructionSentenceEncoder


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper


class MultipleDatasetConstructor:
    """


    """

    def __init__(self, num_iterations=20, base_work_int_id=0, works_per_iteration=1_000_000):
        self.num_iterations = num_iterations
        self.base_work_int_id = base_work_int_id
        self.works_per_iteration = works_per_iteration
        self.checkpoint_file = r"C:\Users\doren\PycharmProjects\CITATION_GRABBER_V2\SENTENCE_ENCODER\works_subfield_checkpoint.json"
        self.model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs32_2024_08_12\checkpoint-124985"
        self.output_directory = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models"


    def merge_filtered_works_data(self):
        base_dir = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER"
        datasets_collected_dir = os.path.join(base_dir, "datasets_collected")
        os.makedirs(datasets_collected_dir, exist_ok=True)

        self.common_authors_file = os.path.join(datasets_collected_dir, "works_common_authors_merged.parquet")
        self.common_titles_file = os.path.join(datasets_collected_dir, "common_title_works_merged.parquet")

        # Process common titles
        print("Processing common titles...")
        self._merge_and_filter_works_data(self.common_titles_file, "works_all_collected_titles_merged_abstract.parquet")

        # Process common authors
        print("Processing common authors...")
        self._merge_and_filter_works_data(self.common_authors_file, "works_all_collected_authors_merged_abstract.parquet")


    def create_subfield_parquets(self):
        # Path to the merged parquet file
        input_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\datasets_collected\works_all_collected_titles_merged_abstract.parquet"

        # Directory to save the subfield parquets
        output_dir = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\datasets_collected\subfield_abstracts"
        os.makedirs(output_dir, exist_ok=True)

        # Read the input parquet file
        df = pd.read_parquet(input_file)

        # Group by subfield
        grouped = df.groupby('subfield_string')

        # Dictionary to store clean_subfield to original subfield mapping
        subfield_mapping = {}

        # Iterate over each subfield and save a separate parquet file
        for subfield, group in tqdm(grouped, desc="Processing subfields"):
            # Clean the subfield name to use as a filename
            clean_subfield = "".join(c if c.isalnum() else "_" for c in str(subfield))
            output_file = os.path.join(output_dir, f"{clean_subfield}.parquet")

            # Save the group as a parquet file
            group.to_parquet(output_file, index=False)

            # Add to the mapping dictionary
            subfield_mapping[clean_subfield] = subfield

            print(f"Saved {len(group)} abstracts for subfield: {subfield}")

        # Save the subfield mapping as a JSON file
        mapping_file = os.path.join(output_dir, "subfield_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(subfield_mapping, f, indent=4)

        print(f"Finished creating parquet files for {len(grouped)} subfields.")
        print(f"Subfield mapping saved to {mapping_file}")

    def _merge_and_filter_works_data(self, filter_file, output_filename):
        # Load work IDs from filter file
        filter_df = pd.read_parquet(filter_file)
        filter_work_ids = set(filter_df['work_id_one']) | set(filter_df['work_id_two'])
        print(f"Loaded {len(filter_work_ids)} unique work IDs from {filter_file}")

        base_dir = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER"
        datasets_collected_dir = os.path.join(base_dir, "datasets_collected")

        all_works_dfs = []
        total_rows_before = 0
        total_rows_after = 0

        # Specify the columns we want to keep
        columns_to_keep = ['work_id', 'abstract_string', 'field_string', 'subfield_string']

        # Find all datasets directories
        dataset_dirs = glob.glob(os.path.join(base_dir, "datasets_*"))

        for dataset_dir in tqdm(dataset_dirs, desc="Processing directories"):
            try:
                last_work_int_id = os.path.basename(dataset_dir).split('_')[-1]
                if "collected" in last_work_int_id or "triplets" in last_work_int_id:
                    continue

                works_file = os.path.join(dataset_dir, "works_all_collected.parquet")
                if os.path.exists(works_file):
                    # Read only the specified columns
                    df = pd.read_parquet(works_file, columns=columns_to_keep)
                    total_rows_before += len(df)

                    # Filter rows based on work_id
                    df_filtered = df[df['work_id'].isin(filter_work_ids)]
                    total_rows_after += len(df_filtered)

                    all_works_dfs.append(df_filtered)
                    print(f"Loaded and filtered {len(df_filtered)} rows from {works_file}")

            except Exception as e:
                print(f"Error processing {dataset_dir}: {e}")

        print(f"Total rows before filtering: {total_rows_before}")
        print(f"Total rows after filtering: {total_rows_after}")

        # Concatenate all dataframes
        combined_df = pd.concat(all_works_dfs, ignore_index=True)

        # Remove duplicates based on work_id
        combined_df.drop_duplicates(subset='work_id', keep='first', inplace=True)
        final_rows = len(combined_df)

        print(f"Final rows after deduplication: {final_rows}")
        print(f"Removed {total_rows_after - final_rows} duplicate rows")

        # Save the consolidated dataframe
        output_file = os.path.join(datasets_collected_dir, output_filename)
        combined_df.to_parquet(output_file, index=False)
        print(f"Saved filtered and merged works data to {output_file}")


    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            last_work_int_id = checkpoint.get("last_work_int_id", self.base_work_int_id)
        else:
            last_work_int_id = self.base_work_int_id
        print(f"Last work_int_id: {last_work_int_id}")
        return last_work_int_id

    def save_checkpoint(self, work_int_id):
        with open(self.checkpoint_file, 'w') as f:
            json.dump({"last_work_int_id": work_int_id}, f)

    def run_total_scores_recalculator(self):
        base_dir = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER"
        datasets_collected_dir = os.path.join(base_dir, "datasets_collected")
        os.makedirs(datasets_collected_dir, exist_ok=True)

        self.unigram_data_file = os.path.join(self.output_directory, "unigram_data.arrow")
        self.bigram_data_file = os.path.join(self.output_directory, "bigram_data.arrow")

        # Load ngram data
        unigrams_df, bigrams_df = self.load_ngrams()

        # Find all datasets directories
        dataset_dirs = glob.glob(os.path.join(base_dir, "datasets_*"))
        for i, dataset_dir in enumerate(dataset_dirs):
            try:
                self.datasets_directory = dataset_dir
                self.works_all_collected_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")

                files_to_process = {
                    "works_common_authors": os.path.join(self.datasets_directory, "works_common_authors.parquet"),
                    "common_title_works": os.path.join(self.datasets_directory, "common_title_works.parquet"),
                    "works_knn_search": os.path.join(self.datasets_directory, "works_knn_search.parquet"),
                    "hard_negatives_pool": os.path.join(self.datasets_directory, "hard_negatives_pool.parquet"),
                    "works_augmented_data": os.path.join(self.datasets_directory, "works_augmented_data.parquet")
                }

                works_filtered_df = pd.read_parquet(self.works_all_collected_file)

                for file_name, file_path in files_to_process.items():
                    if os.path.exists(file_path):
                        print(f"Processing {file_name}")
                        df = pd.read_parquet(file_path)

                        all_work_ids = set(df['work_id_one']) | set(df['work_id_two'])
                        work_details = self.fetch_work_details(all_work_ids, works_filtered_df)

                        pairs = list(zip(df['work_id_one'], df['work_id_two']))
                        vectorized_unigrams, vectorized_bigrams, vectorized_fields, vectorized_subfields = self.process_and_vectorize_common_elements(
                            work_details, pairs)

                        insert_data = self.create_insert_data(pairs, work_details, vectorized_unigrams,
                                                              vectorized_bigrams,
                                                              vectorized_fields, vectorized_subfields)

                        insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)

                        # Update the total_score in the original DataFrame
                        df['total_score'] = [item['total_score'] for item in insert_data]

                        # Calculate and print mean and median scores
                        mean_score = df['total_score'].mean()
                        median_score = df['total_score'].median()
                        print(f"For {file_name}:")
                        print(f"  Mean total_score: {mean_score:.4f}")
                        print(f"  Median total_score: {median_score:.4f}")

                        # Save the updated DataFrame with "_new" appended to the filename
                        new_file_path = file_path.replace('.parquet', '_new.parquet')
                        df.to_parquet(new_file_path, index=False)
                        print(f"Saved updated file: {new_file_path}")
                        print("--------------------")

            except Exception as e:
                print(f"Error processing {self.datasets_directory}: {e}")

        print("Finished recalculating scores for all datasets.")

    def load_ngrams(self):

        unigrams_df = pd.read_feather(self.unigram_data_file)
        bigrams_df = pd.read_feather(self.bigram_data_file)

        return unigrams_df, bigrams_df

    @measure_time
    def process_and_vectorize_common_elements(self, work_details, pairs):
        common_unigrams, common_bigrams, common_fields, common_subfields = self.process_common_elements(work_details,
                                                                                                        pairs)

        vectorized_unigrams = self.vectorized_common_unigrams(common_unigrams)
        vectorized_bigrams = self.vectorized_common_bigrams(common_bigrams)
        vectorized_fields = self.vectorized_common_fields(common_fields)
        vectorized_subfields = self.vectorized_common_subfields(common_subfields)

        return vectorized_unigrams, vectorized_bigrams, vectorized_fields, vectorized_subfields


    @measure_time
    def calculate_total_scores(self, insert_data, unigrams_df, bigrams_df):
        df = pd.DataFrame(insert_data)

        # Vectorized calculation of unigram scores
        df['unigram_score'] = self.vectorized_gram_scores(df['common_uni_grams'], unigrams_df, bigrams_df)

        # Vectorized calculation of bigram scores
        df['bigram_score'] = self.vectorized_gram_scores(df['common_bi_grams'], unigrams_df, bigrams_df)

        # Calculate average gram score
        df['avg_gram_score'] = (df['unigram_score'] + df['bigram_score']) / 2

        scalar_multiplier = 0.05
        df['field_score'] = df.apply(
            lambda row: row['common_field'] * (3.0 + 2 * scalar_multiplier * row['avg_gram_score']), axis=1)
        df['subfield_score'] = df.apply(
            lambda row: row['common_subfield'] * (1.0 + scalar_multiplier * row['avg_gram_score']), axis=1)

        # Vectorized calculation of total score
        df['total_score'] = df['unigram_score'] + df['bigram_score'] + df['field_score'] + df['subfield_score']

        # Convert back to list of dictionaries
        return df.to_dict('records')


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
                    'common_uni_grams': vectorized_unigrams[i],  # Changed from 'common_unigrams'
                    'common_bi_grams': vectorized_bigrams[i],  # Changed from 'common_bigrams'
                    'common_field': bool(vectorized_fields[i]),
                    'common_subfield': bool(vectorized_subfields[i]),
                    'total_score': 0.0,
                    'label': '',
                    'label_int': 0,
                    'p_value': 0.0
                })
        return insert_data

    def create_full_string(self, work):
        return f"{work.get('title_string', '')} {work.get('authors_string', '')} {work.get('field_string', '')} {work.get('subfield_string', '')}"

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
    def get_gram_scores(self, grams, unigrams_df, bigrams_df):
        if not grams:
            return {}

        # Determine if we're dealing with unigrams or bigrams
        is_unigram = 'unigram_type' in unigrams_df.columns

        df = unigrams_df if is_unigram else bigrams_df
        gram_type = "unigram_type" if is_unigram else "bigram_type"

        scores = df[df[gram_type].isin(grams)].set_index(gram_type)['score'].to_dict()

        # Use a default value of 1.0 for any gram not found in the dataframe
        return {gram: float(scores.get(gram, 2.5)) for gram in grams}

    @measure_time
    def vectorized_gram_scores(self, gram_series: pd.Series, unigrams_df, bigrams_df) -> pd.Series:
        # Flatten all sets into a single list
        all_grams = list(set([gram for gram_set in gram_series for gram in gram_set]))

        # Get scores for all grams at once
        scores_dict = self.get_gram_scores(all_grams, unigrams_df, bigrams_df)

        # Create a vectorized function to sum scores
        def sum_scores(gram_set):
            return sum(scores_dict.get(gram, 0.01) for gram in gram_set)

        return gram_series.apply(sum_scores)


    @measure_time
    def calculate_ngram_scores_from_counts(self, df, is_bigram):
        multiplier = 20.0 if is_bigram else 20.0
        df['score'] = np.round(
            multiplier / (np.log((df['count']) + 2) - 1 / np.log(df['count'] + 3) + df['count'] / 100000), 4
        )
        return df

    @measure_time
    def consolidate_works_data(self):
        base_dir = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER"
        datasets_collected_dir = os.path.join(base_dir, "datasets_collected")
        os.makedirs(datasets_collected_dir, exist_ok=True)

        all_works_dfs = []
        total_rows_before = 0
        total_rows_after = 0

        # Find all datasets directories
        dataset_dirs = glob.glob(os.path.join(base_dir, "datasets_*"))

        for i, dataset_dir in enumerate(dataset_dirs):
            try:
                last_work_int_id = os.path.basename(dataset_dir).split('_')[-1]
                if "collected" in last_work_int_id:
                    continue
                if "triplets" in last_work_int_id:
                    continue

                works_file = os.path.join(dataset_dir, "works_all_collected.parquet")
                if os.path.exists(works_file):

                    df = pd.read_parquet(works_file)
                    total_rows_before += len(df)
                    all_works_dfs.append(df)
                    print(f"Loaded {len(df)} rows from {works_file}")

            except Exception as e:
                print("Error: ", e)

        print(f"Total rows before deduplication: {total_rows_before}")

        # Concatenate all dataframes
        combined_df = pd.concat(all_works_dfs, ignore_index=True)

        # Remove duplicates based on work_id
        combined_df.drop_duplicates(subset='work_id', keep='first', inplace=True)
        total_rows_after = len(combined_df)

        print(f"Total rows after deduplication: {total_rows_after}")
        print(f"Removed {total_rows_before - total_rows_after} duplicate rows")

        # Save the consolidated dataframe
        output_file = os.path.join(datasets_collected_dir, "works_all_collected_consolidated.parquet")
        combined_df.to_parquet(output_file, index=False)
        print(f"Saved consolidated works data to {output_file}")

        return combined_df



    def run(self):
        for i in range(self.num_iterations):
            last_work_int_id = self.load_checkpoint()
            current_work_int_id = last_work_int_id + (i * self.works_per_iteration)

            datasets_directory = f"C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\datasets_{current_work_int_id}"
            os.makedirs(datasets_directory, exist_ok=True)

            print(f"Starting iteration {i + 1}/{self.num_iterations}")
            print(f"Processing works from {current_work_int_id} to {current_work_int_id + self.works_per_iteration}")
            # False
            run_params = {
                'load_and_print_data': False,
                'collect_works_for_subfields': True,
                'restructure_common_authors': True,
                'restructure_augmented_data': True,
                'preprocess_and_calculate_ngrams': False,
                'batch_update_ngram_scores': False,
                'create_sentence_embeddings': True,
                'calculate_density_scores': False,
                'build_vector_index': True,
                'generate_training_pairs': True,
                'create_common_title_works': True,
                'generate_all_work_id_pairs_dataset': True,
            }

            encoder = DatasetConstructionSentenceEncoder(
                model_path=self.model_path,
                output_directory=self.output_directory,
                datasets_directory=datasets_directory,
                run_params=run_params,
                num_knn_pairs=self.works_per_iteration,
                num_works_collected=self.works_per_iteration,
                mongo_url="mongodb://localhost:27017/",
                mongo_database_name="CiteGrab",
                mongo_works_collection_name="Works"
            )

            encoder.run()
            encoder.triplets_quality_control_statistics()

            self.save_checkpoint(current_work_int_id + self.works_per_iteration)

            print(f"Completed iteration {i + 1}/{self.num_iterations}")
            print("--------------------------------------------")

    def collect_and_process_triplets(self):
        base_dir = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER"
        datasets_collected_dir = os.path.join(base_dir, "datasets_collected")
        os.makedirs(datasets_collected_dir, exist_ok=True)

        all_triplets = []

        # Find all datasets directories
        dataset_dirs = glob.glob(os.path.join(base_dir, "datasets_*"))

        for i, dataset_dir in enumerate(dataset_dirs):
            try:
                last_work_int_id = os.path.basename(dataset_dir).split('_')[-1]
                if "collected" in last_work_int_id:
                    continue
                if "triplets" in last_work_int_id:
                    continue

                triplets_file = os.path.join(dataset_dir, "triplets.parquet")

                if os.path.exists(triplets_file):
                    # Read the triplets file
                    df = pd.read_parquet(triplets_file)

                    # Save with new name in datasets_collected directory
                    new_file_name = f"triplets_batch_{last_work_int_id}.parquet"
                    new_file_path = os.path.join(datasets_collected_dir, new_file_name)
                    df.to_parquet(new_file_path, index=False)
                    print(f"Saved {new_file_name}")

                    all_triplets.append(df)
            except Exception as e:
                print("Error: ", e)

        # Concatenate all triplets
        combined_triplets = pd.concat(all_triplets, ignore_index=True)

        # Create filtered version
        filtered_triplets = self.filter_triplets(combined_triplets)
        filtered_file_path = os.path.join(datasets_collected_dir, "triplets_filtered.parquet")
        filtered_triplets.to_parquet(filtered_file_path, index=False)
        print(f"Saved filtered triplets to {filtered_file_path}")
        print(f"length of filtered triplets:  {len(filtered_triplets)}")

    def filter_triplets(self, df):
        # Remove duplicates
        df = df.drop_duplicates()

        # Sort by max_pos_neg_distance in descending order
        df = df.sort_values('max_pos_neg_distance', ascending=False)

        # Keep only top 500 million rows or all if less than 10 million
        df = df.head(500_000_000)

        return df


    def merge_common_title_works(self):
        return self._merge_files("common_title_works.parquet", "common_title_works_merged.parquet")

    def merge_works_knn_search(self):
        return self._merge_files("works_knn_search.parquet", "works_knn_search_merged.parquet")

    def merge_works_common_authors(self):
        return self._merge_files("works_common_authors.parquet", "works_common_authors_merged.parquet")

    def merge_triplet_work_ids_only(self):
        return self._merge_files("triplet_work_ids_only_*.parquet", "triplet_work_ids_only_merged.parquet")

    def _merge_files(self, file_pattern, output_filename):
        base_dir = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER"
        datasets_collected_dir = os.path.join(base_dir, "datasets_collected")
        os.makedirs(datasets_collected_dir, exist_ok=True)

        all_dfs = []
        total_rows = 0

        # Find all datasets directories
        dataset_dirs = glob.glob(os.path.join(base_dir, "datasets_*"))

        for dataset_dir in tqdm(dataset_dirs, desc=f"Processing {file_pattern}"):
            try:
                last_work_int_id = os.path.basename(dataset_dir).split('_')[-1]
                if "collected" in last_work_int_id or "triplets" in last_work_int_id:
                    continue

                file_path = os.path.join(dataset_dir, file_pattern)
                matching_files = glob.glob(file_path)

                for file in matching_files:
                    df = pd.read_parquet(file)
                    # Keep only 'work_id_one' and 'work_id_two' columns
                    df = df[['work_id_one', 'work_id_two']]
                    total_rows += len(df)
                    all_dfs.append(df)
                    print(f"Loaded {len(df)} rows from {file}")

            except Exception as e:
                print(f"Error processing {dataset_dir}: {e}")

        print(f"Total rows before merging: {total_rows}")

        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Remove duplicates
        combined_df.drop_duplicates(inplace=True)
        total_rows_after = len(combined_df)

        print(f"Total rows after merging and deduplication: {total_rows_after}")
        print(f"Removed {total_rows - total_rows_after} duplicate rows")

        # Save the merged dataframe
        output_file = os.path.join(datasets_collected_dir, output_filename)
        combined_df.to_parquet(output_file, index=False)
        print(f"Saved merged data to {output_file}")

        return combined_df

    def merge_all_files(self):

        print("\nMerging triplet_work_ids_only files...")
        # self.merge_triplet_work_ids_only()

        print("Merging common_title_works files...")
        self.merge_common_title_works()

        print("\nMerging works_common_authors files...")
        self.merge_works_common_authors()

        print("\nMerging works_knn_search files...")
        self.merge_works_knn_search()

        print("\nAll files have been merged successfully.")


if __name__ == "__main__":
    multiple_constructor = MultipleDatasetConstructor(num_iterations=10, base_work_int_id=46098504,
                                                      works_per_iteration=3_000_000)

    # "C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\datasets_47598419"
    # up to 50_000_000 (2million to cover).
    # multiple_constructor.collect_and_process_triplets()
    # consolidated_df = multiple_constructor.consolidate_works_data()

    # multiple_constructor.merge_filtered_works_data()
    multiple_constructor.create_subfield_parquets()
    # multiple_constructor.collect_and_process_triplets()

    # multiple_constructor.run()

    # Saved batch 8 to C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\datasets_collected\common_authors_batches\common_title_works_triplets_num_8.parquet
    #  24%|██▍       | 6042938/25385559 [6:02:11<18:01:29, 298.09it/s]
    # Saving batch 9 to C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\datasets_collected\common_authors_batches\common_title_works_triplets_num_9.parquet

    # Extracted D:\openalex-snapshot\data\works\updated_date=2024-07-28\part_015.gz to D:\openalex-extracted-data\openalex-snapshot\data\works\updated_date=2024-07-28\part_015