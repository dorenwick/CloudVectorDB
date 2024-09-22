import gc

import polars as pl

import os


class NGramLoader:
    """

        We wish to make trigrams from bigrams, in the following way:
        whenever we get two bigrams in an abstract in a sequence [n, n+1] and [n+1, n+2], I wish
        for you to add the trigram to a postgresql schema and table for trigrams:

        whenever we get two bigrams in an abstract in a sequence [n, n+1] and [n+1, n+2] and [n+2, n+3],
        we should add the four gram to a postgresql schema and table for four grams.

        Here is what we will do:

        We will go to this database and this schema: datasets_key_phrase

        and create a table called trigrams_from_bigrams

            db_params = {
            "dbname": "CitationData",
            "user": "postgres",
            "password": "Cl0venh00f$$",
            "host": "localhost",
            "port": 5432
        }

        what this trigrams_from_bigrams will contain, is insertions of two bigrams, whenever they are found in a sequence:
        [n, n+1] and [n+1, n+2] in a abstract_string

        we make a fourgrams_from_bigrams table as well, with the same idea.

        each table will have a field_score, subfield_score, topic_score column.

        these will be vectors with the id's from each column. we will have to create our own id's up for this.

        currently we have:

      these are the field_ids.

      we will need you to connect to the CiteGrab db in Mongodb, with the Subfields collection and Topics collection,
      and use the display_name , subfields_int_id or display_name, topics_int_id to create id2labels for each of these subfields and topics
      schemas, and then make a table called topic_id in the openalex_topics_concepts table and a subfield_id table as well,
      that has two columns:
      subfield_id, display_name

      topic_id, display_name.

      Then, we will use these to create the vector lists of integers for the field_score, subfield_score, topic_score columns

      in the trigrams_from_bigrams and fourgrams_from_bigrams tables

      we will also want a bigrams table that constructs similar lists.

    Second filtering options:

    for bigrams, we want to calculate tf_icf scores, which is term frequency, inverse concept freqency scores.
    where concepts are the 26 or 27 concepts in the field vector (cant remember how many dimensions it has).

    we want to score the words by taking 1 and multiplying it by N**2 where N is the number of zero integers in the field_score vector.
    and we will divide by the count of the ngram, or rather log(tf) where tf=count , or term frequency.

    so tf_icf = N**2 / log(tf).

    """


    def __init__(self):
        self.chunk_size = 1_000_000
        self.filtered_chunks_dir = r"E:\NGRAMS\filtered_chunks"
        self.file_paths = {
            "full_string_bigrams": r"E:\NGRAMS\full_string_bigrams.parquet",
            "full_string_unigrams": r"E:\NGRAMS\full_string_unigrams.parquet",
            "short_bigrams": r"E:\NGRAMS\short_bigrams.parquet",
            "short_unigrams": r"E:\NGRAMS\short_unigrams.parquet"
        }
        self.bigram_counts = {}

    def get_output_file(self, input_file):
        base_name = os.path.basename(input_file)
        return os.path.join(os.path.dirname(input_file), f"filtered_{base_name}")

    def process_file(self, input_file, is_bigram):
        output_file = self.get_output_file(input_file)
        chunk_files = []
        chunk_number = 0

        print(f"Processing {input_file}...")

        # Create a LazyFrame
        lf = pl.scan_parquet(input_file)

        # Get total number of rows
        total_rows = lf.select(pl.count()).collect().item()

        # Process in chunks
        for start in range(0, total_rows, self.chunk_size):
            end = min(start + self.chunk_size, total_rows)

            # Collect only the current chunk
            chunk = lf.slice(start, self.chunk_size).collect()

            # Define a function to count non-zero integers in field_score
            def count_non_zero(field_score):
                return sum(1 for score in field_score if score != 0)

            # Apply filters
            if is_bigram:
                filtered_chunk = chunk.filter(
                    (pl.col("count") >= 40) &
                    (pl.col("count") <= 100_000) &
                    (pl.col("field_count").map_elements(count_non_zero) <= 6)
                )
            else:
                filtered_chunk = chunk.filter(
                    (pl.col("count") >= 100) &
                    (pl.col("count") <= 200_000) &
                    (pl.col("field_count").map_elements(count_non_zero) <= 20)
                )

            print("length filtered chunk: ", len(filtered_chunk))
            # Print top 10 and bottom 10 ngrams
            print(f"\nTop 10 ngrams for chunk {chunk_number}:")
            print(filtered_chunk.sort("count", descending=True).select(["ngram", "count", "field_count"]).head(10))


            print(f"\nBottom 10 ngrams for chunk {chunk_number}:")
            print(filtered_chunk.sort("count").select(["ngram", "count", "field_count"]).head(10))

            chunk_file = os.path.join(self.filtered_chunks_dir, f"chunk_{chunk_number}.parquet")
            filtered_chunk.write_parquet(chunk_file)
            gc.collect()
            chunk_files.append(chunk_file)
            chunk_number += 1

            print(f"Processed rows {start} to {end} out of {total_rows}")

        # Concatenate all chunks
        pl.concat([pl.scan_parquet(f) for f in chunk_files]).sink_parquet(output_file)

        # Clean up temporary chunk files
        for f in chunk_files:
            os.remove(f)

        print(f"Finished processing. Output saved to {output_file}")
        return output_file

    def process_all_files(self):
        os.makedirs(self.filtered_chunks_dir, exist_ok=True)
        processed_files = []

        for file_type, input_file in self.file_paths.items():
            is_bigram = "bigram" in file_type
            output_file = self.process_file(input_file, is_bigram)
            processed_files.append(output_file)

        return processed_files

    def print_top_10(self, file_path):
        try:
            df = pl.read_parquet(file_path)
            print(f"\nTop 10 rows of {os.path.basename(file_path)}:")
            print(df.head(10))
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

    def load_and_index_bigrams(self):
        filtered_bigrams_file = self.get_output_file(self.file_paths["full_string_bigrams"])
        print(f"Loading and indexing bigrams from {filtered_bigrams_file}...")

        # Read the filtered bigrams file
        df = pl.read_parquet(filtered_bigrams_file)

        # Create a dictionary with ngram as key and count as value
        self.bigram_counts = dict(zip(df['ngram'].to_list(), df['count'].to_list()))

        print(f"Indexed {len(self.bigram_counts)} bigrams.")

    def lookup_bigrams(self, bigrams):
        if not self.bigram_counts:
            print("Bigrams have not been loaded. Loading now...")
            self.load_and_index_bigrams()

        results = {}
        for bigram in bigrams:
            count = self.bigram_counts.get(bigram, 0)
            results[bigram] = count
            print(f"'{bigram}': {count}")

        return results


if __name__ == "__main__":
    loader = NGramLoader()

    # Process all files if they haven't been processed yet
    if not os.path.exists(loader.get_output_file(loader.file_paths["full_string_bigrams"])):
        loader.process_all_files()

    # Load and index bigrams
    loader.load_and_index_bigrams()

    # Look up specific bigrams
    bigrams_to_lookup = [
        "art gallery", "gallery theorem", "algebraic topology", "noam chomsky",
        "fixed point", "hairy ball", "epistemic injustice", "principal ideal"
    ]
    loader.lookup_bigrams(bigrams_to_lookup)

    # "E:\NGRAMS\filtered_full_string_bigrams.parquet"