import polars as pl
import pandas as pd
import psycopg2
from psycopg2 import sql
import spacy
from collections import defaultdict
from tqdm import tqdm
import os

class NGramKeyphraseProcessor:
    def __init__(self, datasets_directory, pg_config, mongo_config, unigram_keyphrase_file, bigram_keyphrase_file):
        self.datasets_directory = datasets_directory
        self.pg_config = pg_config
        self.mongo_config = mongo_config
        self.unigram_keyphrase_file = unigram_keyphrase_file
        self.bigram_keyphrase_file = bigram_keyphrase_file
        self.nlp = spacy.load("en_core_web_sm")
        self.field_map = {}
        self.subfield_map = {}
        self.topic_map = {}
        self.unigram_keyphrases = {}
        self.bigram_keyphrases = {}

    def load_mapping_data(self):
        conn_params = {k: v for k, v in self.pg_config.items() if k != 'schema'}
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        for table, map_dict in [('field_id', self.field_map),
                                ('subfield_id', self.subfield_map),
                                ('topic_id', self.topic_map)]:
            cursor.execute(sql.SQL("""
                SELECT {id_column}, display_name 
                FROM {schema}.{table}
            """).format(
                id_column=sql.Identifier(f"{table[:-3]}_id"),
                schema=sql.Identifier(self.pg_config['schema']),
                table=sql.Identifier(table)
            ))
            for row in cursor.fetchall():
                map_dict[row[1]] = row[0]  # map display_name to id

        cursor.close()
        conn.close()


    def load_keyphrases(self):
        # Load unigrams
        unigram_df = pd.read_parquet(self.unigram_keyphrase_file)
        self.unigram_keyphrases = dict(zip(unigram_df['ngram'], unigram_df['count']))

        # Load bigrams
        bigram_df = pd.read_parquet(self.bigram_keyphrase_file)
        self.bigram_keyphrases = dict(zip(bigram_df['ngram'], bigram_df['count']))

        print(f"Loaded {len(self.unigram_keyphrases)} unigrams and {len(self.bigram_keyphrases)} bigrams")

    def load_keyphrases_test(self):
        unigrams_of_interest = ['cohomology', 'functor', 'chomsky', 'panpsychism']
        bigrams_of_interest = ['algebraic topology', 'art gallery', 'gallery theorem', 'epistemic injustice']

        # Load unigrams
        unigram_df = pd.read_parquet(self.unigram_keyphrase_file)

        print("Unigrams of interest:")
        for unigram in unigrams_of_interest:
            row = unigram_df[unigram_df['ngram'] == unigram]
            if not row.empty:
                print(f"  {unigram}:")
                print(f"    Count: {row['count'].values[0]}")
                print(f"    Field counts: {row['field_count'].values[0]}")
            else:
                print(f"  {unigram} not found in the dataset.")

        # Load bigrams
        bigram_df = pd.read_parquet(self.bigram_keyphrase_file)

        print("\nBigrams of interest:")
        for bigram in bigrams_of_interest:
            row = bigram_df[bigram_df['ngram'] == bigram]
            if not row.empty:
                print(f"  {bigram}:")
                print(f"    Count: {row['count'].values[0]}")
                print(f"    Field counts: {row['field_count'].values[0]}")
            else:
                print(f"  {bigram} not found in the dataset.")

    def process_abstract(self, abstract):
        doc = self.nlp(abstract)
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        unigrams = tokens
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        trigrams = [f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}" for i in range(len(tokens) - 2)]
        fourgrams = [f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]} {tokens[i + 3]}" for i in range(len(tokens) - 3)]

        return unigrams, bigrams, trigrams, fourgrams

    def find_keyphrases(self, tokens):
        keyphrases = []
        i = 0
        while i < len(tokens) - 1:
            if f"{tokens[i]} {tokens[i + 1]}" in self.bigram_keyphrases:
                keyphrase = [tokens[i], tokens[i + 1]]
                j = i + 2
                while j < len(tokens) and (f"{tokens[j - 1]} {tokens[j]}" in self.bigram_keyphrases or tokens[j] in self.unigram_keyphrases):
                    keyphrase.append(tokens[j])
                    j += 1
                keyphrases.append(" ".join(keyphrase))
                i = j
            elif tokens[i] in self.unigram_keyphrases:
                keyphrases.append(tokens[i])
                i += 1
            else:
                i += 1
        return keyphrases

    def process_works(self):
        self.load_mapping_data()
        self.load_keyphrases()

        works_file = os.path.join(self.datasets_directory, "works_batch_0.parquet")
        df = pl.read_parquet(works_file)
        print("Schema of works_batch_0.parquet:")
        print(df.schema)

        ngram_counts = {
            'unigram': defaultdict(lambda: [0] * (max(self.field_map.values()) + 1)),
            'bigram': defaultdict(lambda: [0] * (max(self.field_map.values()) + 1)),
            'trigram': defaultdict(lambda: [0] * (max(self.field_map.values()) + 1)),
            'fourgram': defaultdict(lambda: [0] * (max(self.field_map.values()) + 1))
        }

        def process_row(row):
            if row['abstract_string']:
                unigrams, bigrams, trigrams, fourgrams = self.process_abstract(row['abstract_string'])
                keyphrases = self.find_keyphrases(unigrams)

                field_id = self.field_map.get(row['field_string'], -1)
                subfield_id = self.subfield_map.get(row['subfield_string'], -1)
                topic_id = self.topic_map.get(row['topic'], -1)

                for gram_type, grams in [('unigram', unigrams), ('bigram', bigrams),
                                         ('trigram', trigrams), ('fourgram', fourgrams)]:
                    for gram in grams:
                        if field_id != -1:
                            ngram_counts[gram_type][gram][field_id] += 1
                        if subfield_id != -1:
                            ngram_counts[gram_type][gram][subfield_id] += 1
                        if topic_id != -1:
                            ngram_counts[gram_type][gram][topic_id] += 1

            return pl.Series([keyphrases])

        print("Processing works...")
        df_with_keyphrases = df.with_columns(pl.struct(['abstract_string', 'field_string', 'subfield_string', 'topic'])
                                             .map_elements(process_row)
                                             .alias('keyphrases'))

        print("Saving updated DataFrame with keyphrases...")
        df_with_keyphrases.write_parquet(os.path.join(self.datasets_directory, "works_with_keyphrases.parquet"))

        print("Creating and saving n-gram tables...")
        for gram_type, counts in ngram_counts.items():
            ngram_df = pl.DataFrame({
                'ngram': list(counts.keys()),
                'count': [sum(v) for v in counts.values()],
                'field': [v[:max(self.field_map.values()) + 1] for v in counts.values()],
                'subfield': [v[max(self.field_map.values()) + 1:max(self.subfield_map.values()) + 1] for v in counts.values()],
                'topic': [v[max(self.subfield_map.values()) + 1:] for v in counts.values()]
            })
            ngram_df.write_parquet(os.path.join(self.datasets_directory, f"{gram_type}_table.parquet"))

    def run(self):
        print("Starting NGramKeyphraseProcessor...")
        self.process_works()
        print("NGramKeyphraseProcessor completed.")


if __name__ == "__main__":
    datasets_directory = "E:\HugeDatasetBackup\WORKS_BATCHES"

    # PostgreSQL configuration
    pg_config = {
        'host': 'localhost',
        'database': 'CitationData',
        'user': 'postgres',
        'password': 'Cl0venh00f$$',
        'port': 5432,
        'schema': 'openalex_topics_concepts'  # Updated to the correct schema
    }

    # MongoDB configuration
    mongo_config = {
        'url': "mongodb://localhost:27017/",
        'database': "OpenAlex",
        'collection': "Works"
    }

    unigram_keyphrase_file = r"E:\NGRAMS\filtered_full_string_unigrams.parquet"
    bigram_keyphrase_file = r"E:\NGRAMS\filtered_full_string_bigrams.parquet"

    processor = NGramKeyphraseProcessor(
        datasets_directory,
        pg_config,
        mongo_config,
        unigram_keyphrase_file,
        bigram_keyphrase_file
    )
    # processor.load_keyphrases_test()
    processor.run()