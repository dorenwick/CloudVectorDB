import os
import json
import torch
import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from span_marker import SpanMarkerModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict



#     E:\HugeDatasetBackup\DATA_CITATION_GRABBER\datasets_collected\works_combined_data_batch_0.parquet
#     Keep this here just to tell me what the thing is located at.

class AbstractDataConstructionMultiGPU():
    """


    This class will implement a few things:

    We will be given a directory with the following:
    parquet files of name format: works_combined_data_batch_0.parquet where # is a number. The numbers range from 0, 1, ..., 143 (maybe higher)
    and are around 700mb each (5gb~ as dataframes in pandas).

    they have format:

    ({
        'work_id': work_id,
        "works_int_id": works_int_id,
        'has_abstract': has_abstract,  # New field
        'title': title,
        'authors_string': authors_string,
        'combined_string': combined_string,
        'field': field,
        'subfield': subfield,
        'topic': topic,

    })

    where all of these entries are strings type.

    We are going to make a directory called /workspace on a linux machine with multiple gpu's.
    The idea here will be to augment each of these files as one that has encodings produced for it.
    
    we make four new columns:
    
    keywords_title:
    
    this column will have a list of dictionaries that contain the output of a ner model in the span marker library
    where we run the keywords classifier on the title
    
    acronyms_title:
    
    this column will have a list of dictionaries that contain the output of a ner mdoel in the span marker library.
    where we run the acronyms classifier on the title

    keywords_abstract:

    this column will have a list of dictionaries that contain the output of a ner model in the span marker library
    where we run the keywords classifier on the abstract_string.

    acronyms_abstract:

    this column will have a list of dictionaries that contain the output of a ner model in the span marker library.
    where we run the acronyms classifier on the title

    I wish to also make an unigrams and bigrams parquet file where we collect unigrams and bigrams over the joint:

    full_string = title + authors_string + abstract_string

    full_string_unigrams.parquet has columns (unigram_type, unigram_count, unigram_smoothed_score, unigram_ctf_idf_score, field_count)
    full_string_bigrams.parquet has columns (bigram_type, bigram_count, bigram_smoothed_score, bigram_ctf_idf_score, field_count)
    field_int_map.json will contain id2label, label2id keys that are dictionaries of field map to int and vice versa.
    Here we have a 26 dimensional vector containing integers, that are counts for each time the ngram occurs in the field.

    we will also want to generate a
    short_unigrams.parquet
    and
    short_bigrams.parquet
    file, which will contain the (unigram_type, unigram_count, unigram_score, field_count) as well, but these will only be
    for unigrams and bigrams in short_string = f{title} + {authors_string}.strip().

    keywords_data.parquet will contain all keywords we have ever detected. We will process this in a postprocessing step.
    This will have every keyword and the work_id it's associated with, and also have columns for score, and char_start, char_end.

    acronyms_data.parquet will contain all acronyms we have ever detected. We will process this in a postprocessing step.
    This will have every acronym and the work_id it's associated with, and also have columns for score, and char_start, char_end.

    Now, I want to use encoders to do the following:

    create embeddings using the snowflake-xs model on these strings:

    self.model_path = r"C:\Users\doren\.cache\huggingface\hub\models--Snowflake--snowflake-arctic-embed-xs\snapshots\86a07656cc240af5c7fd07bac2f05baaafd60401"

    we shall also use this classifier to create embeddings for:

    full_string = f"{title} {authors_string} {field} {subfield} {topic} {keywords}".strip()

    topic_string = f"{title} {field} {subfield} {topic} {keywords}".strip()

    abstract_string = f"{abstract_string}"

    for topic_string, make precision="binary" as well as non precision ones.
    for abstract_string, make precision="binary" as well as non precision ones as well.
    for full_string, only make full embeddings.

    We shall batch these in parquet files of size 100_000 each.
    Do not specify batch size when we do this, let sentence transformers decide for itself.
    we will be using A100, and cuda 12 version, probably 12.4.

    We will be creating full_string, topic_string columns, and we will be creating the following five columns as well:

    full_string_embeddings, abstract_string_embeddings, abstract_string_embeddings_binary, topic_string_embeddings, topic_string_embeddings_binary

    So, in summary, we load the parquet file with structure:

    ({
        'work_id': work_id,
        "works_int_id": works_int_id,
        'has_abstract': has_abstract,  # New field
        'title': title,
        'authors_string': authors_string,
        'combined_string': combined_string,
        'field': field,
        'subfield': subfield,
        'topic': topic,
    })

    and we save them with structure:

    ({
        'work_id': str,
        'works_int_id': str,
        'has_abstract': str,
        'title': str,
        'authors_string': str,
        'combined_string': str,
        'field': str,
        'subfield': str,
        'topic': str,
        'keywords_title': List[Dict],  # List of dictionaries from NER model output
        'acronyms_title': List[Dict],  # List of dictionaries from NER model output
        'keywords_abstract': List[Dict],  # List of dictionaries from NER model output
        'acronyms_abstract': List[Dict],  # List of dictionaries from NER model output
        'full_string': str,
        'topic_string': str,
        'full_string_embeddings': List[float],
        'abstract_string_embeddings': List[float],
        'abstract_string_embeddings_binary': List[float],
        'topic_string_embeddings': List[float],
        'topic_string_embeddings_binary': List[float]
    })

    List of parquet files to be saved:

    1. Processed abstract data files:
       - works_abstracts_processed_batch_#.parquet (where # is the batch number)

    2. N-gram files:
       - full_string_unigrams.parquet
       - full_string_bigrams.parquet
       - short_unigrams.parquet
       - short_bigrams.parquet

    3. Keyword and acronym data files:
       - keywords_data.parquet
       - acronyms_data.parquet

    4. Embedding files:
       - embeddings_batch_#.parquet (where # is the batch number, containing 100,000 entries each)

    5. Field mapping file:
       - field_int_map.json

    This structure and list of files cover all the processed data and additional files mentioned in the class description.


    Because we have memory constraints, I want you to at first process the following:

    Do a loop where we run through and build the four ngram parquet files.
    We can do that by loading up our pre-computed batch parquet files, and by creating the field_int_map json, and loading it as
    a dictionary.

    We ideally wish to use counters to build up the ngrams data before saving it all as parquets. Possibly we will use dictionaries instead/as well.


    """

    def __init__(self, input_dir: str,
                 output_dir: str,
                 keyphrase_model_path: str,
                 acronym_model_path: str,
                 embedding_model_path: str,
                 batch_size: int = 100_000):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models using provided paths
        self.keyphrase_model = SpanMarkerModel.from_pretrained(keyphrase_model_path).to(self.device)
        self.acronym_model = SpanMarkerModel.from_pretrained(acronym_model_path).to(self.device)
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)

        # Load or create field_int_map
        self.field_int_map = self.load_or_create_field_int_map()
        print("self.field_int_map: ", self.field_int_map)

        # Initialize counters for n-grams
        self.full_unigrams = Counter()
        self.full_bigrams = Counter()
        self.short_unigrams = Counter()
        self.short_bigrams = Counter()

    def load_or_create_field_int_map(self) -> Dict[str, Dict[str, int]]:
        field_int_map_path = os.path.join(self.output_dir, "field_int_map.json")
        if os.path.exists(field_int_map_path):
            with open(field_int_map_path, 'r') as f:
                return json.load(f)
        else:
            # Create field_int_map based on the provided mapping
            id2label = {
                0: "Economics, Econometrics and Finance",
                1: "Materials Science",
                2: "Environmental Science",
                3: "Medicine",
                4: "Psychology",
                5: "Dentistry",
                6: "Business, Management and Accounting",
                7: "Engineering",
                8: "Biochemistry, Genetics and Molecular Biology",
                9: "Agricultural and Biological Sciences",
                10: "Energy",
                11: "Earth and Planetary Sciences",
                12: "Health Professions",
                13: "Chemistry",
                14: "Chemical Engineering",
                15: "Social Sciences",
                16: "Pharmacology, Toxicology and Pharmaceutics",
                17: "Arts and Humanities",
                18: "Mathematics",
                19: "Immunology and Microbiology",
                20: "Veterinary",
                21: "Decision Sciences",
                22: "Nursing",
                23: "Physics and Astronomy",
                24: "Neuroscience",
                25: "Computer Science"
            }
            label2id = {v: k for k, v in id2label.items()}
            field_int_map = {"id2label": id2label, "label2id": label2id}
            with open(field_int_map_path, 'w') as f:
                json.dump(field_int_map, f)
            return field_int_map

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Extract keywords and acronyms
        batch['keywords_title'] = self.extract_entities(batch['title'].tolist(), self.keyphrase_model)
        batch['acronyms_title'] = self.extract_entities(batch['title'].tolist(), self.acronym_model)
        batch['keywords_abstract'] = self.extract_entities(batch['combined_string'].tolist(), self.keyphrase_model)
        batch['acronyms_abstract'] = self.extract_entities(batch['combined_string'].tolist(), self.acronym_model)

        # Create full_string and topic_string
        batch['full_string'] = batch.apply(lambda
                                               row: f"{row['title']} {row['authors_string']} {row['field']} {row['subfield']} {row['topic']} {' '.join([k['span'] for k in row['keywords_title'] + row['keywords_abstract']])}".strip(),
                                           axis=1)
        batch['topic_string'] = batch.apply(lambda
                                                row: f"{row['title']} {row['field']} {row['subfield']} {row['topic']} {' '.join([k['span'] for k in row['keywords_title'] + row['keywords_abstract']])}".strip(),
                                            axis=1)

        # Generate embeddings
        batch['full_string_embeddings'] = self.generate_embeddings(batch['full_string'].tolist())
        batch['abstract_string_embeddings'] = self.generate_embeddings(batch['combined_string'].tolist())
        batch['abstract_string_embeddings_binary'] = self.generate_embeddings(batch['combined_string'].tolist(),
                                                                              binary=True)
        batch['topic_string_embeddings'] = self.generate_embeddings(batch['topic_string'].tolist())
        batch['topic_string_embeddings_binary'] = self.generate_embeddings(batch['topic_string'].tolist(), binary=True)

        return batch

    def extract_entities(self, texts: List[str], model: SpanMarkerModel) -> List[List[Dict]]:
        return model.predict(texts)

    def generate_embeddings(self, texts: List[str], binary: bool = False) -> np.ndarray:
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        if binary:
            embeddings = (embeddings > 0).float()
        return embeddings.cpu().numpy()

    def update_ngram_counters(self, batch: pd.DataFrame):
        for _, row in batch.iterrows():
            full_text = f"{row['title']} {row['authors_string']} {row['combined_string']}".lower()
            short_text = f"{row['title']} {row['authors_string']}".lower()

            full_words = full_text.split()
            short_words = short_text.split()

            self.full_unigrams.update(full_words)
            self.full_bigrams.update(zip(full_words[:-1], full_words[1:]))
            self.short_unigrams.update(short_words)
            self.short_bigrams.update(zip(short_words[:-1], short_words[1:]))

    def process_file(self, file_path: str):
        df = pd.read_parquet(file_path)
        processed_batches = []

        for i in tqdm(range(0, len(df), self.batch_size), desc=f"Processing {os.path.basename(file_path)}"):
            batch = df.iloc[i:i + self.batch_size]
            processed_batch = self.process_batch(batch)
            processed_batches.append(processed_batch)
            self.update_ngram_counters(processed_batch)

        processed_df = pd.concat(processed_batches, ignore_index=True)
        output_file = os.path.join(self.output_dir, f"processed_{os.path.basename(file_path)}")
        processed_df.to_parquet(output_file, index=False)

    def save_ngram_data(self):
        def save_counter(counter: Counter, file_name: str):
            df = pd.DataFrame([(k, v) for k, v in counter.items()], columns=['ngram', 'count'])
            df['smoothed_score'] = np.log1p(df['count'])
            df['ctf_idf_score'] = df['count'] * np.log(len(counter) / df['count'])
            df['field_count'] = [np.zeros(26, dtype=int) for _ in range(len(df))]  # Placeholder for field counts
            df.to_parquet(os.path.join(self.output_dir, file_name), index=False)

        save_counter(self.full_unigrams, "full_string_unigrams.parquet")
        save_counter(self.full_bigrams, "full_string_bigrams.parquet")
        save_counter(self.short_unigrams, "short_unigrams.parquet")
        save_counter(self.short_bigrams, "short_bigrams.parquet")

    def run(self):
        input_files = [f for f in os.listdir(self.input_dir) if f.endswith('.parquet')]

        # Use multiprocessing to process files in parallel
        with Pool(cpu_count()) as pool:
            pool.map(partial(self.process_file), [os.path.join(self.input_dir, f) for f in input_files])

        self.save_ngram_data()


if __name__ == "__main__":
    input_dir = "/path/to/input/directory"
    output_dir = "/path/to/output/directory"
    keyphrase_model_path = "/path/to/keyphrase/model"
    acronym_model_path = "/path/to/acronym/model"
    embedding_model_path = "/path/to/embedding/model"

    processor = AbstractDataConstructionMultiGPU(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path,
        acronym_model_path=acronym_model_path,
        embedding_model_path=embedding_model_path
    )
    processor.run()


















