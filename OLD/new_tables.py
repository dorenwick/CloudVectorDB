import pymongo
import psycopg2
from psycopg2 import sql


class DatabaseMapper:
    def __init__(self):
        # MongoDB database information
        self.mongo_url = "mongodb://localhost:27017/"
        self.mongo_database_name = "CiteGrab"

        # PostgreSQL database information
        self.pg_host = "localhost"
        self.pg_database = "CitationData"
        self.pg_user = "postgres"
        self.pg_password = "Cl0venh00f$$"
        self.pg_port = 5432
        self.pg_schema = "openalex_topics_concepts"

    def connect_mongo(self):
        client = pymongo.MongoClient(self.mongo_url)
        db = client[self.mongo_database_name]
        return db

    def connect_postgres(self):
        conn = psycopg2.connect(
            host=self.pg_host,
            database=self.pg_database,
            user=self.pg_user,
            password=self.pg_password,
            port=self.pg_port
        )
        return conn

    def create_schema(self):
        pg_conn = self.connect_postgres()
        pg_cursor = pg_conn.cursor()

        pg_cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
            sql.Identifier(self.pg_schema)
        ))

        pg_conn.commit()
        pg_cursor.close()
        pg_conn.close()

    def create_mapping_tables(self):
        mongo_db = self.connect_mongo()
        pg_conn = self.connect_postgres()
        pg_cursor = pg_conn.cursor()

        # Create subfield_id table
        pg_cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.subfield_id (
            subfield_id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL
        )
        """).format(sql.Identifier(self.pg_schema)))

        # Create topic_id table
        pg_cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.topic_id (
            topic_id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL
        )
        """).format(sql.Identifier(self.pg_schema)))

        # Create field_id table
        pg_cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.field_id (
            field_id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL
        )
        """).format(sql.Identifier(self.pg_schema)))

        # Populate subfield_id table
        subfields = mongo_db.Subfields.find({}, {"subfields_int_id": 1, "display_name": 1})
        for subfield in subfields:
            pg_cursor.execute(sql.SQL("""
            INSERT INTO {}.subfield_id (subfield_id, display_name) 
            VALUES (%s, %s) ON CONFLICT DO NOTHING
            """).format(sql.Identifier(self.pg_schema)),
                              (subfield["subfields_int_id"], subfield["display_name"]))

        # Populate topic_id table
        topics = mongo_db.Topics.find({}, {"topics_int_id": 1, "display_name": 1})
        for topic in topics:
            pg_cursor.execute(sql.SQL("""
            INSERT INTO {}.topic_id (topic_id, display_name) 
            VALUES (%s, %s) ON CONFLICT DO NOTHING
            """).format(sql.Identifier(self.pg_schema)),
                              (topic["topics_int_id"], topic["display_name"]))

        # Populate field_id table
        fields = mongo_db.Fields.find({}, {"fields_int_id": 1, "display_name": 1})
        for field in fields:
            pg_cursor.execute(sql.SQL("""
            INSERT INTO {}.field_id (field_id, display_name) 
            VALUES (%s, %s) ON CONFLICT DO NOTHING
            """).format(sql.Identifier(self.pg_schema)),
                              (field["fields_int_id"], field["display_name"]))

        pg_conn.commit()
        pg_cursor.close()
        pg_conn.close()

    def create_ngram_tables(self):
        pg_conn = self.connect_postgres()
        pg_cursor = pg_conn.cursor()

        # Create trigrams_from_bigrams table
        pg_cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.trigrams_from_bigrams (
            id SERIAL PRIMARY KEY,
            trigram TEXT NOT NULL,
            count INTEGER NOT NULL,
            field_score INTEGER[] NOT NULL,
            subfield_score INTEGER[] NOT NULL,
            topic_score INTEGER[] NOT NULL
        )
        """).format(sql.Identifier(self.pg_schema)))

        # Create fourgrams_from_bigrams table
        pg_cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.fourgrams_from_bigrams (
            id SERIAL PRIMARY KEY,
            fourgram TEXT NOT NULL,
            count INTEGER NOT NULL,
            field_score INTEGER[] NOT NULL,
            subfield_score INTEGER[] NOT NULL,
            topic_score INTEGER[] NOT NULL
        )
        """).format(sql.Identifier(self.pg_schema)))

        pg_conn.commit()
        pg_cursor.close()
        pg_conn.close()

    def run_all(self):
        print("Creating schema...")
        self.create_schema()
        print("Creating mapping tables...")
        self.create_mapping_tables()
        print("Creating n-gram tables...")
        self.create_ngram_tables()
        print("All tables created successfully in the openalex_topics_concepts schema.")


if __name__ == "__main__":
    mapper = DatabaseMapper()
    mapper.run_all()