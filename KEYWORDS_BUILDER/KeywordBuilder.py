import polars as pl
import psycopg2
from psycopg2 import sql
import spacy
from collections import defaultdict
from tqdm import tqdm
import os


class NGramKeyphraseProcessor:
    """

    NGramKeyphraseProcessor.

    E:\HugeDatasetBackup\WORKS_BATCHES\works_batch_0.parquet

    We will test on this dataset.




    """

    def __init__(self, datasets_directory, pg_config, unigram_keyphrase_file, bigram_keyphrase_file):
        self.datasets_directory = datasets_directory
        self.pg_config = pg_config
        self.unigram_keyphrase_file = unigram_keyphrase_file
        self.bigram_keyphrase_file = bigram_keyphrase_file
        self.nlp = spacy.load("en_core_web_sm")
        self.field_map = {}
        self.subfield_map = {}
        self.topic_map = {}
        self.unigram_keyphrases = set()
        self.bigram_keyphrases = set()

    def load_mapping_data(self):
        conn = psycopg2.connect(**self.pg_config)
        cursor = conn.cursor()

        for table, map_dict in [('field_id', self.field_map),
                                ('subfield_id', self.subfield_map),
                                ('topic_id', self.topic_map)]:
            cursor.execute(sql.SQL("SELECT * FROM {}.{}").format(
                sql.Identifier(self.pg_config['schema']), sql.Identifier(table)))
            for row in cursor.fetchall():
                map_dict[row[1]] = row[0]  # map display_name to id

        conn.close()

    def load_keyphrases(self):
        self.unigram_keyphrases = set(pl.read_parquet(self.unigram_keyphrase_file)['unigram'])
        self.bigram_keyphrases = set(pl.read_parquet(self.bigram_keyphrase_file)['bigram'])

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
                while j < len(tokens) and (f"{tokens[j - 1]} {tokens[j]}" in self.bigram_keyphrases or tokens[
                    j] in self.unigram_keyphrases):
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

        works_file = os.path.join(self.datasets_directory, "works_all_collected.parquet")
        df = pl.read_parquet(works_file)

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

        df_with_keyphrases = df.with_columns(pl.struct(['abstract_string', 'field_string', 'subfield_string', 'topic'])
                                             .map_elements(process_row)
                                             .alias('keyphrases'))

        # Save the updated DataFrame with keyphrases
        df_with_keyphrases.write_parquet(os.path.join(self.datasets_directory, "works_with_keyphrases.parquet"))

        # Create and save n-gram tables
        for gram_type, counts in ngram_counts.items():
            ngram_df = pl.DataFrame({
                'ngram': list(counts.keys()),
                'count': [sum(v) for v in counts.values()],
                'field': [v[:max(self.field_map.values()) + 1] for v in counts.values()],
                'subfield': [v[max(self.field_map.values()) + 1:max(self.subfield_map.values()) + 1] for v in
                             counts.values()],
                'topic': [v[max(self.subfield_map.values()) + 1:] for v in counts.values()]
            })
            ngram_df.write_parquet(os.path.join(self.datasets_directory, f"{gram_type}_table.parquet"))

    def run(self):
        self.process_works()