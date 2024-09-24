import os
import random
import gc
import polars as pl
from tqdm import tqdm
import numpy as np

class AugmentData:
    def __init__(self, datasets_directory, ngrams_directory):
        self.datasets_directory = datasets_directory
        self.ngrams_directory = ngrams_directory

    def print_memory_usage(self, location):
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage at {location}: {memory_info.rss / 1024 / 1024:.2f} MB")

    def create_augmented_data(self, generate_all_augmentations):
        print("Creating augmented data...")
        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        augmented_df = pl.read_parquet(works_file)

        gc.collect()

        # Load unigram scores
        unigrams_file = os.path.join(self.ngrams_directory, "unigram_data.parquet")
        unigrams_df = pl.read_parquet(unigrams_file)
        unigram_scores_dict = dict(zip(unigrams_df['unigram_type'], unigrams_df['score']))

        self.print_memory_usage(f"memory usage before we generate augmentations")

        def create_augmented_strings(row):
            title_string = row['title_string'] or ""
            authors_string = row['authors_string'] or ""
            field_string = row['field_string'] or ""
            subfield_string = row['subfield_string'] or ""
            author_names = row['author_names'] or []

            full_string = f"{title_string} {authors_string} {field_string} {subfield_string}".strip()

            # Get unigram scores
            unigram_scores = {word: unigram_scores_dict.get(word.lower(), 2.5) for word in full_string.split()}

            # Sort unigrams by score
            sorted_unigrams = sorted(unigram_scores.items(), key=lambda x: x[1], reverse=True)

            # Get top scoring unigrams
            top_unigrams = [word for word, _ in sorted_unigrams[:3]]

            # Define all possible augmentations
            augmentations = [
                ('full_title', lambda: title_string),
                ('full_title_field', lambda: f"{title_string} {field_string}"),
                ('author_field', lambda: f"{author_names[0] if author_names else ''} {field_string}"),
                ('all_authors_field', lambda: f"{' '.join(author_names)} {field_string}"),
                ('one_author_field_subfield', lambda: f"{author_names[0] if author_names else ''} {field_string} {subfield_string}"),
                ('two_authors_field_subfield', lambda: f"{' '.join(author_names[:2])} {field_string} {subfield_string}"),
                ('two_authors_field', lambda: f"{' '.join(author_names[:2])} {field_string}"),
                ('full_title_field_subfield', lambda: f"{title_string} {field_string} {subfield_string}"),
                ('all_authors_field_subfield', lambda: f"{' '.join(author_names)} {field_string} {subfield_string}"),
                ('field', lambda: field_string),
                ('field_subfield', lambda: f"{field_string} {subfield_string}"),
                ('top_unigram', lambda: top_unigrams[0] if top_unigrams else ''),
                ('top_two_unigrams', lambda: ' '.join(top_unigrams[:2]) if len(top_unigrams) >= 2 else ''),
                ('top_three_unigrams', lambda: ' '.join(top_unigrams[:3]) if len(top_unigrams) >= 3 else ''),
                ('top_unigram_field_subfield', lambda: f"{top_unigrams[0] if top_unigrams else ''} {field_string} {subfield_string}"),
                ('authors_no_initials', lambda: ' '.join([name for name in author_names if len(name) > 2]))
            ]

            # Filter augmentations based on available data
            valid_augmentations = [
                aug for aug in augmentations
                if (('title' not in aug[0] or title_string) and
                    ('author' not in aug[0] or authors_string) and
                    ('field' not in aug[0] or field_string) and
                    ('subfield' not in aug[0] or subfield_string))
            ]

            # If no valid augmentations, use a default
            if not valid_augmentations:
                return [{'full_string': full_string, 'augmented_string': "Science", 'augmentation_type': 'default'}]

            if generate_all_augmentations:
                # Generate all valid augmentations
                augmented_strings = []
                for augmentation_type, augmentation_func in valid_augmentations:
                    augmented_string = augmentation_func().strip()
                    if augmented_string and augmented_string != full_string:
                        augmented_strings.append({
                            'full_string': full_string,
                            'augmented_string': augmented_string,
                            'augmentation_type': augmentation_type
                        })
                return augmented_strings
            else:
                # Select an augmentation at random (original behavior)
                augmentation_type, augmentation_func = random.choice(valid_augmentations)
                augmented_string = augmentation_func().strip()

                if not augmented_string or augmented_string == full_string:
                    words = full_string.split()
                    augmented_string = random.choice(words) if words else "Science"

                return [{'full_string': full_string, 'augmented_string': augmented_string,
                         'augmentation_type': augmentation_type}]

        gc.collect()

        # Apply the augmentation to each row
        augmented_df = augmented_df.with_columns([
            pl.struct(['title_string', 'authors_string', 'field_string', 'subfield_string', 'author_names'])
            .map_elements(create_augmented_strings)
            .alias('augmented')
        ]).explode('augmented').with_columns([
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

        self.print_memory_usage(f"memory usage after we generate augmentations")

        return augmented_df

    def restructure_augmented_data(self, generate_all_augmentations, filter_high_similarity=0.01):
        self.create_augmented_data(generate_all_augmentations=generate_all_augmentations)
        augmented_data_file = os.path.join(self.datasets_directory, "works_augmented_data.parquet")
        print("Filtering augmented data file...")

        # Read the parquet file
        df = pl.read_parquet(augmented_data_file)

        # TODO: Test.
        df = df[:10_000]

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
            unigrams_one = row['full_string_one'].lower().split() if row['full_string_one'] else []
            bigrams_one = [f"{unigrams_one[i]} {unigrams_one[i + 1]}" for i in range(len(unigrams_one) - 1)]

            # Process full_string_two
            unigrams_two = row['full_string_two'].lower().split() if row['full_string_two'] else []
            bigrams_two = [f"{unigrams_two[i]} {unigrams_two[i + 1]}" for i in range(len(unigrams_two) - 1)]

            # Find common elements
            common_unigrams = list(set(unigrams_one) & set(unigrams_two))
            common_bigrams = list(set(bigrams_one) & set(bigrams_two))

            # Return a dictionary with lists and booleans
            return {
                "common_unigrams": common_unigrams,
                "common_bigrams": common_bigrams,
                "common_field": True,
                "common_subfield": True
            }

        # Debug: Print schema of filtered_df
        print(f"Debug - filtered_df schema: {filtered_df.schema}")

        processed_df = filtered_df.with_columns([
            pl.struct(['full_string_one', 'full_string_two'])
            .map_elements(
                process_common_elements,
                return_dtype=pl.Struct([
                    pl.Field("common_unigrams", pl.List(pl.Utf8)),
                    pl.Field("common_bigrams", pl.List(pl.Utf8)),
                    pl.Field("common_field", pl.Boolean),
                    pl.Field("common_subfield", pl.Boolean)
                ])
            ).alias('processed')
        ])

        # Debug: Print schema after map_elements
        print(f"Debug - After map_elements schema: {processed_df.schema}")

        processed_df = processed_df.with_columns([
            pl.col('processed').struct.field('common_unigrams').alias('common_unigrams'),
            pl.col('processed').struct.field('common_bigrams').alias('common_bigrams'),
            pl.col('processed').struct.field('common_field').alias('common_field'),
            pl.col('processed').struct.field('common_subfield').alias('common_subfield')
        ]).drop('processed')

        # Debug: Print final schema
        print(f"Debug - Final processed_df schema: {processed_df.schema}")

        # Debug: Print a few rows of the processed DataFrame
        print("Debug - First few rows of processed_df:")
        print(processed_df.head())

        unigrams_df, bigrams_df = self.load_ngrams()

        # Calculate total scores
        insert_data = self.calculate_total_scores(processed_df.to_dicts(), unigrams_df, bigrams_df)

        # Convert insert_data back to DataFrame
        result_df = pl.DataFrame(insert_data)

        # Filter out top percentage of pairs based on total_score
        if filter_high_similarity > 0:
            threshold_score = result_df['total_score'].quantile(1 - filter_high_similarity)
            result_df = result_df.filter(pl.col('total_score') < threshold_score)
            print(f"Filtered out top {filter_high_similarity:.2%} of pairs with total_score >= {threshold_score:.4f}")

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

    def load_ngrams(self):
        unigrams_df = pl.read_parquet(os.path.join(self.ngrams_directory, "unigram_data.parquet"))
        bigrams_df = pl.read_parquet(os.path.join(self.ngrams_directory, "bigram_data.parquet"))