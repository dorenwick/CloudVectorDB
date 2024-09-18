import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class KeyPhraseFromAbstract:
    def __init__(self, input_dir, output_dir, keywords_file):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.keywords_file = keywords_file
        self.keywords = self.load_keywords()

    def load_keywords(self):
        df = pd.read_parquet(self.keywords_file)
        keywords = {1: set(), 2: set(), 3: set(), 4: set()}
        for _, row in df.iterrows():
            phrase = row['keyphrase'] if 'keyphrase' in df.columns else row['keyword']
            if isinstance(phrase, str):  # Ensure phrase is a string
                words = phrase.lower().split()
                n = len(words)
                if 1 <= n <= 4:
                    keywords[n].add(phrase)
        return keywords

    def extract_ngrams(self, text, n):
        if not isinstance(text, str):
            return []
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    def extract_keywords(self, text):
        if not isinstance(text, str):
            return []
        keywords = []
        for n in range(1, 5):
            ngrams = self.extract_ngrams(text, n)
            keywords.extend([gram for gram in ngrams if gram in self.keywords[n]])
        return keywords

    def merge_adjacent_keywords(self, keywords):
        merged = []
        i = 0
        while i < len(keywords):
            if i < len(keywords) - 2:
                w1, w2, w3 = keywords[i].split(), keywords[i + 1].split(), keywords[i + 2].split()
                if len(w1) + len(w2) + len(w3) <= 4 and w1[-1] == w2[0] and w2[-1] == w3[0]:
                    merged.append(' '.join(w1 + w2[1:] + w3[1:]))
                    i += 3
                    continue
            if i < len(keywords) - 1:
                w1, w2 = keywords[i].split(), keywords[i + 1].split()
                if len(w1) + len(w2) <= 4 and w1[-1] == w2[0]:
                    merged.append(' '.join(w1 + w2[1:]))
                    i += 2
                    continue
            merged.append(keywords[i])
            i += 1
        return merged

    def process_file(self, file_path):
        df = pd.read_parquet(file_path)

        df['keywords_title'] = df['title'].apply(lambda x: self.extract_keywords(x))
        df['keywords_title'] = df['keywords_title'].apply(self.merge_adjacent_keywords)

        df['keywords_abstract'] = df['abstract_string'].apply(lambda x: self.extract_keywords(x))
        df['keywords_abstract'] = df['keywords_abstract'].apply(self.merge_adjacent_keywords)

        return df

    def save_processed_file(self, df, original_filename):
        output_path = os.path.join(self.output_dir, f"processed_{original_filename}")
        df.to_parquet(output_path, index=False)

    def process_all_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for filename in tqdm(os.listdir(self.input_dir)):
            if filename.endswith('.parquet'):
                file_path = os.path.join(self.input_dir, filename)
                processed_df = self.process_file(file_path)
                self.save_processed_file(processed_df, filename)

    def run(self):
        print("Starting keyword extraction process...")
        self.process_all_files()
        print("Keyword extraction completed successfully.")


if __name__ == "__main__":
    input_dir = "/path/to/input/parquet/files"
    output_dir = "/path/to/output/parquet/files"
    keywords_file = "/path/to/keywords_full.parquet"

    extractor = KeyPhraseFromAbstract(input_dir, output_dir, keywords_file)
    extractor.run()