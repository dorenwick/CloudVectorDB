import pandas as pd
import psycopg2
from psycopg2 import sql


class KeyWordDatasetConstructor:
    def __init__(self, db_params, parquet_file_paths):
        self.db_params = db_params
        self.parquet_file_paths = parquet_file_paths
        self.conn = None
        self.cursor = None

    def connect_to_db(self):
        self.conn = psycopg2.connect(**self.db_params)
        self.cursor = self.conn.cursor()

    def close_db_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def get_filtered_ngrams(self):
        ngram_data = []
        for table in ['unigram', 'bigram', 'trigram', 'ngram_4plus']:
            query = sql.SQL("""
                SELECT keyphrase, count, max_score, mean_score 
                FROM datasets_key_phrase.{} 
            """).format(sql.Identifier(f"filtered_{table}"))

            self.cursor.execute(query)
            ngram_data.extend(self.cursor.fetchall())
        return pd.DataFrame(ngram_data, columns=['keyphrase', 'count', 'max_score', 'mean_score'])

    def get_high_scoring_bigrams(self, min_count=10):
        self.cursor.execute("""
            SELECT ngram, count, ctf_idf_score
            FROM datasets_key_phrase.full_string_bigrams
            WHERE count >= %s
            ORDER BY ctf_idf_score DESC
            LIMIT 10000
        """, (min_count,))
        bigram_data = self.cursor.fetchall()
        return pd.DataFrame(bigram_data, columns=['keyphrase', 'count', 'ctf_idf_score'])

    def get_openalex_keywords(self):
        self.cursor.execute("""
            SELECT keyword, topics, subfield, field, domain
            FROM openalex_topics_concepts.openalex_keywords
        """)
        keyword_data = self.cursor.fetchall()
        return pd.DataFrame(keyword_data, columns=['keyword', 'topics', 'subfield', 'field', 'domain'])

    def get_openalex_concepts(self):
        self.cursor.execute("""
            SELECT display_name, level
            FROM openalex_topics_concepts.openalex_concepts
            WHERE level >= 2
        """)
        concept_data = self.cursor.fetchall()
        return pd.DataFrame(concept_data, columns=['concept', 'level'])

    def get_keywords_from_parquet(self):
        df = pd.read_parquet(self.parquet_file_paths['keywords'])
        unigrams = df[(df['entity'].str.split().str.len() == 1) & (df['score'] > 0.94)]
        bigrams = df[(df['entity'].str.split().str.len() == 2) & (df['score'] > 0.92)]
        trigrams = df[(df['entity'].str.split().str.len() == 3) & (df['score'] > 0.88)]
        fourgrams = df[(df['entity'].str.split().str.len() >= 4) & (df['score'] > 0.8)]
        return pd.concat([unigrams, bigrams, trigrams, fourgrams])

    def construct_dataset(self):
        self.connect_to_db()

        filtered_ngrams = self.get_filtered_ngrams()
        high_scoring_bigrams = self.get_high_scoring_bigrams()
        openalex_keywords = self.get_openalex_keywords()
        openalex_concepts = self.get_openalex_concepts()
        keywords_from_parquet = self.get_keywords_from_parquet()

        self.close_db_connection()

        # Combine all dataframes
        # You may need to adjust this depending on the exact structure you want
        combined_df = pd.concat([
            filtered_ngrams,
            high_scoring_bigrams,
            openalex_keywords,
            openalex_concepts,
            keywords_from_parquet
        ], axis=0, ignore_index=True)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['keyphrase'])

        return combined_df

    def save_dataset(self, output_path):
        dataset = self.construct_dataset()
        dataset.to_parquet(output_path)
        print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    db_params = {
        "dbname": "CitationData",
        "user": "postgres",
        "password": "Cl0venh00f$$",
        "host": "localhost",
        "port": 5432
    }
    parquet_file_paths = {
        "keywords": "C:\\Users\\doren\\PycharmProjects\\CloudVectorDB\\keywords_data.parquet"
    }
    output_path = "C:\\Users\\doren\\PycharmProjects\\CloudVectorDB\\keywords_full.parquet"

    constructor = KeyWordDatasetConstructor(db_params, parquet_file_paths)
    constructor.save_dataset(output_path)