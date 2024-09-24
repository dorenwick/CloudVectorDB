

import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


class NgramScoring:
    """
    TF (Term Frequency):

    Measures how frequently a term appears in a document.
    Formula: TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)

    IDF (Inverse Document Frequency):

    Measures how important a term is across the entire corpus.
    Formula: IDF(t) = log(N / DF(t))
    Where:
    N = Total number of documents in the corpus
    DF(t) = Number of documents containing term t


    TF-IDF:

    Combines TF and IDF to calculate the final score.
    Formula: TF-IDF(t,d) = TF(t,d) * IDF(t)


    We wish to make this class take in a list of parquet files generated from AbstractDataConstructionMultiProcessing.
    What we will do is generate a variation of the tf-idf amoung the following lines.

    We want to score the word by how often it appears in the corpus.
    So, we take the total number of counts of all ngrams for n=k, and divide by the count of the term, and the count
    oh how many topics it appeared in.

    Low score: Very frequent words appearing in almost every topic
    examples: "the", "is", "in", "to", "a", "that", "for"
    Medium score: Frequent words appearing in a few specific topics
    examples: "algorithm", "theorem", "hypothesis", "quantum", "gene"
    Medium score: Less frequent words appearing across many topics
    examples: "analysis", "model", "theory", "system", "structure"
    High score: Infrequent words appearing in very few topics
    examples: "panpsychism", "qualia", "bosons", "chirality", "functor"

    So, we want our score to have some log( N**k / (tf * cf ))) where k is a constant parameter, with default k=1.
    this will be an argument to the method and class.

    And we will make it so that the class argument contains the directory where the parquet files are located,
    and a list of parquet files themselves. The schema of the parquet file shall be ngram (n=1,2,3,4) and count
    and field_count

    Here we will need a system that reads through these parquet files, and generates subfield_score and topic_score
    which are vectors of integers counting how many times they occur in each topic.

    So, we will do that first in order to get the counting. Then, we will generate the scores of the ngrams and such.
    I have detailed how to build the vectors and count the topic/subfield occurrences in the AbstractDataConstructionMultiProcessing
    class..we have shown how to do it with the field vector there. This time you will do the exact same thing except for
    specific keywords from a given subset parquet file (like the ones we make from that class), such as the medium_filtered parquet files.

    So, to generate the scores, we will go through each abstract_string in those batch parquet files and then load up the subset parquet files
    and add columns for subfield_count and topic_count, and go through all of those.

    TODO: We will also want to generate scoring for the non filtered ngrams at some point as well, but those will have to be
        based off field_count, as there are far too many.

    TODO: We will also require that we have specific scoring for short unigrams and short bigrams based off
        log( N**k / (tf * fc ))) where fc = field_count

    """


    def __init__(self, input_dir, parquet_files, k=1):
        self.input_dir = input_dir
        self.parquet_files = parquet_files
        self.k = k
        self.total_ngrams = {}
        self.ngram_data = {}
        self.subfield_map = None
        self.topic_map = None

    def load_mapping_files(self):
        # Load subfield and topic mapping files
        subfield_map_path = os.path.join(self.input_dir, 'subfield_int_map.json')
        topic_map_path = os.path.join(self.input_dir, 'topic_int_map.json')

        if os.path.exists(subfield_map_path):
            with open(subfield_map_path, 'r') as f:
                self.subfield_map = json.load(f)

        if os.path.exists(topic_map_path):
            with open(topic_map_path, 'r') as f:
                self.topic_map = json.load(f)

    def process_ngram_files(self):
        for file in self.parquet_files:
            file_path = os.path.join(self.input_dir, file)
            df = pd.read_parquet(file_path)
            n = len(df['ngram'].iloc[0].split())  # Determine n-gram length

            self.total_ngrams[n] = df['count'].sum()

            for _, row in df.iterrows():
                ngram = row['ngram']
                count = row['count']
                field_count = row['field_count']

                self.ngram_data[ngram] = {
                    'count': count,
                    'field_count': field_count,
                    'subfield_count': np.zeros(len(self.subfield_map['id2label'])),
                    'topic_count': np.zeros(len(self.topic_map['id2label']))
                }

    def process_abstract_files(self, abstract_files):
        for file in tqdm(abstract_files, desc="Processing abstract files"):
            file_path = os.path.join(self.input_dir, file)
            df = pd.read_parquet(file_path)

            for _, row in df.iterrows():
                abstract = row['abstract_string']
                subfield = row['subfield']
                topic = row['topic']

                words = abstract.lower().split()
                for n in range(1, 5):  # For unigrams to 4-grams
                    for i in range(len(words) - n + 1):
                        ngram = ' '.join(words[i:i + n])
                        if ngram in self.ngram_data:
                            self.ngram_data[ngram]['subfield_count'][self.subfield_map['label2id'][subfield]] += 1
                            self.ngram_data[ngram]['topic_count'][self.topic_map['label2id'][topic]] += 1

    def calculate_scores(self):
        N = len(self.ngram_data)
        for ngram, data in self.ngram_data.items():
            n = len(ngram.split())
            tf = data['count'] / self.total_ngrams[n]
            cf = sum(data['field_count'] > 0)
            score = np.log(N * self.k / (tf * cf))
            self.ngram_data[ngram]['score'] = score

    def save_results(self, output_file):
        results = []
        for ngram, data in self.ngram_data.items():
            results.append({
                'ngram': ngram,
                'score': data['score'],
                'count': data['count'],
                'field_count': data['field_count'].tolist(),
                'subfield_count': data['subfield_count'].tolist(),
                'topic_count': data['topic_count'].tolist()
            })

        df = pd.DataFrame(results)
        df.to_parquet(output_file, index=False)

    def run(self, abstract_files, output_file):
        self.load_mapping_files()
        self.process_ngram_files()
        self.process_abstract_files(abstract_files)
        self.calculate_scores()
        self.save_results(output_file)

# Usage example:
# scorer = NgramScoring(input_dir='path/to/input', parquet_files=['unigrams.parquet', 'bigrams.parquet', ...])
# scorer.run(abstract_files=['abstracts_1.parquet', 'abstracts_2.parquet', ...], output_file='ngram_scores.parquet')