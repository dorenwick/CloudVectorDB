import csv
import psycopg2
from transformers import AutoTokenizer

class DatasetTopicClassification:
    """
    TODO: we have added title_similarity into our xml_data table (as seen here):
                    # Insert results into the database
                   #  work_id, file_name, title, top_concept_id, top_concept, top_concept_score,
                     all_concepts, topic_name, subfield_name, field_name, domain_name,
                     work_display_name, title_similarity

    TODO: Now, we want to filter for rows that have title_similarity of 0.99 or higher. This will ensure that we are filtering for
        #  data that didn't make an error in the topic stuff.

    Here I want to use xml_data and paragraph_data class to build another table called topic_classification_dataset in datasets schema.
    We will build a table that has paragraph_id, paragraph_text, topic, subfield, field, domain, concepts (possibly with scores, take what we have from the paragraph_data table).
    We will create a method that counts how many paragraphs we have of each topic, subfield, field and domain and then prints the results.

    We want to construct to modify the create_topic_classification_dataset method to do the following:

    We will also concatenate paragraphs together until n > 512 tokens, but n- 1 < 512 tokens, and create a dataset
    doing that. We will do this by using the fact that work_id, paragraph_id in paragraph_data table are related
    in the following way:

    work_id will be same for all paragraphs in a work, and paragraph_id rsplit('_') gives us an integer that is
    the index of the paragraph. For this reason, we will build training text docs in the following way:
    Iterate over the rows in paragraph_data, and, for each work_id, run prajjawal1-bert-tiny tokenizer on each
    paragraph. Now, we add paragraphs to a sequence until we get n > 512 tokens, and then remove the last one to go back
    under 512 tokens. Then, we start over at the removed paragraph, doing the same thing, until we reach a row with a new
    work_id. Going this way, we build larger text_sequences, (and we keep the label given to us via the work_id, since
    the topic, subfield, field, domain label is the same all paragraphs in a work_id. So, recreate this
    method so that we get the datasets.topic_classification_dataset but paragraph_text will now have packed paragraphs
    in them. And use the bert-tiny tokenizer.

    P_https://openalex.org/W101716136_5


    """

    def __init__(self):
        # xml_directory = "D:\\ACADEMIC\\XML ENGLISH"
        # datasets_directory = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER"
        self.output_dir = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER"

        # Connect to the PostgreSQL database
        self.pg_conn = psycopg2.connect(
            host="localhost",
            database="CitationData",
            user="postgres",
            password="Cl0venh00f$$"
        )
        self.pg_cur = self.pg_conn.cursor()

    def create_topic_classification_dataset(self):
        # Create the 'topic_classification_dataset' table in the 'datasets' schema
        self.pg_cur.execute("""
            CREATE TABLE IF NOT EXISTS datasets.topic_classification_dataset (
                paragraph_id VARCHAR,
                paragraph_text TEXT,
                topic VARCHAR,
                subfield VARCHAR,
                field VARCHAR,
                domain VARCHAR,
                concepts VARCHAR
            )
        """)
        self.pg_conn.commit()

        # Load the prajjwal1/bert-tiny tokenizer
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

        # Iterate over the rows in paragraph_data
        self.pg_cur.execute("""
            SELECT pd.work_id, pd.paragraph_id, pd.paragraph_text, xd.topic_name, xd.subfield_name, xd.field_name, xd.domain_name, xd.all_concepts
            FROM citation_data.paragraph_data pd
            JOIN citation_data.xml_data xd ON pd.work_id = xd.work_id
            WHERE xd.title_similarity > 0.98
            ORDER BY pd.work_id, CAST(split_part(pd.paragraph_id, '_', 3) AS INTEGER)
        """)
        rows = self.pg_cur.fetchall()

        current_work_id = None
        paragraph_text = ""
        paragraph_ids = []
        topic = None
        subfield = None
        field = None
        domain = None
        concepts = None

        for row in rows:
            work_id, paragraph_id, paragraph, topic_name, subfield_name, field_name, domain_name, all_concepts = row

            if work_id != current_work_id:
                # Insert the previous paragraph_text into the dataset
                if paragraph_text:
                    self.pg_cur.execute("""
                        INSERT INTO datasets.topic_classification_dataset (paragraph_id, paragraph_text, topic, subfield, field, domain, concepts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (','.join(paragraph_ids), paragraph_text, topic, subfield, field, domain, concepts))
                    self.pg_conn.commit()

                # Reset the variables for the new work_id
                current_work_id = work_id
                paragraph_text = ""
                paragraph_ids = []
                topic = topic_name
                subfield = subfield_name
                field = field_name
                domain = domain_name
                concepts = all_concepts

            # Tokenize the current paragraph
            tokens = tokenizer.tokenize(paragraph)

            # Check if adding the current paragraph exceeds the token limit
            if len(tokenizer.encode(paragraph_text + ' ' + ' '.join(tokens))) > 512:
                # Insert the previous paragraph_text into the dataset
                self.pg_cur.execute("""
                    INSERT INTO datasets.topic_classification_dataset (paragraph_id, paragraph_text, topic, subfield, field, domain, concepts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (','.join(paragraph_ids), paragraph_text, topic, subfield, field, domain, concepts))
                self.pg_conn.commit()

                # Start a new paragraph_text with the current paragraph
                paragraph_text = paragraph
                paragraph_ids = [paragraph_id]
            else:
                # Append the current paragraph to the existing paragraph_text
                paragraph_text += ' ' + paragraph
                paragraph_ids.append(paragraph_id)

        # Insert the last paragraph_text into the dataset
        if paragraph_text:
            self.pg_cur.execute("""
                INSERT INTO datasets.topic_classification_dataset (paragraph_id, paragraph_text, topic, subfield, field, domain, concepts)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (','.join(paragraph_ids), paragraph_text, topic, subfield, field, domain, concepts))
            self.pg_conn.commit()

    def count_paragraphs_by_category(self):
        categories = ['topic', 'subfield', 'field', 'domain']

        for category in categories:
            self.pg_cur.execute(f"""
                SELECT {category}, COUNT(*) AS count
                FROM datasets.topic_classification_dataset
                GROUP BY {category}
                ORDER BY count DESC
            """)
            result = self.pg_cur.fetchall()

            print(f"Paragraph counts by {category}:")
            for row in result:
                print(f"{row[0]}: {row[1]}")
            print()

    def save_dataset_to_csv(self):
        output_file = f"{self.output_dir}/topic_classification_dataset.csv"

        self.pg_cur.execute("""
            SELECT paragraph_id, paragraph_text, topic, subfield, field, domain, concepts
            FROM datasets.topic_classification_dataset
        """)
        rows = self.pg_cur.fetchall()

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['paragraph_id', 'paragraph_text', 'topic', 'subfield', 'field', 'domain', 'concepts'])
            writer.writerows(rows)

        print(f"Dataset saved to CSV: {output_file}")

    def save_text_classification_datasets_to_csv(self):
        categories = ['topic', 'subfield', 'field']

        for category in categories:
            output_file = f"{self.output_dir}/topic_classification_dataset_{category}.csv"

            self.pg_cur.execute(f"""
                SELECT ROW_NUMBER() OVER () AS document_id, paragraph_text, {category}
                FROM datasets.topic_classification_dataset
            """)
            rows = self.pg_cur.fetchall()

            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['document_id', 'paragraph_text', category])
                writer.writerows(rows)

            print(f"Text classification dataset for {category} saved to CSV: {output_file}")



    def close_connection(self):
        self.pg_cur.close()
        self.pg_conn.close()


if __name__ == "__main__":
    dataset_topic_classification = DatasetTopicClassification()

    dataset_topic_classification.create_topic_classification_dataset()
    dataset_topic_classification.count_paragraphs_by_category()
    dataset_topic_classification.save_dataset_to_csv()
    dataset_topic_classification.save_text_classification_datasets_to_csv()

    dataset_topic_classification.close_connection()