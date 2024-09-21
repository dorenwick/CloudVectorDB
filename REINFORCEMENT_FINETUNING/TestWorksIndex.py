import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm


class CloudDatasetReinforcementFinetuning:
    """
    TODO: to make the missed_data.parquet I want you to walk over mongodb Works collection,
    and pick out the work_id and and work object for works with cited_by_count that is 25 or greater.
    use projection to make it easier.
    Then once we have found 1_000 works, I want


    """

    def __init__(self, index_path, model_path, work_data_path):
        self.index_path = index_path
        self.model_path = model_path
        self.work_data_path = work_data_path
        self.index = None
        self.model = None
        self.work_data = None

    def load_resources(self):
        print("Loading index...")
        self.index = faiss.read_index(self.index_path)

        print("Loading model...")
        self.model = SentenceTransformer(self.model_path)

        print("Loading work data...")
        self.work_data = pl.read_parquet(self.work_data_path)

    def generate_queries(self, row):
        title = row['title']
        field = row['field']
        subfield = row['subfield']
        author_names = row['author_names']

        title_words = title.split()
        trigram = " ".join(title_words[:min(3, len(title_words))])

        queries = [
            f"{title} {field}",
            f"{trigram} {field} {subfield}",
            f"{' '.join(author_names[:2])} {field}",
            f"{author_names[0] if author_names else ''} {field} {subfield}",
            f"{title} {' '.join(author_names)} {field} {subfield}"  # full query
        ]

        return queries

    def search_similar_works(self, query_vector, k=20):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices[0]

    def find_missed_works(self):
        missed_data = []

        for row in tqdm(self.work_data.iter_rows(named=True), total=len(self.work_data), desc="Processing works"):
            work_id = row['work_id']
            works_int_id = row['works_int_id']
            queries = self.generate_queries(row)

            results = []
            query_strings = []

            for query in queries:
                query_vector = self.model.encode([query])[0]
                similar_indices = self.search_similar_works(query_vector)
                found = works_int_id in similar_indices

                results.append(int(found))
                query_strings.append(query)

            if 0 in results:
                missed_data.append({
                    'work_id': work_id,
                    'works_int_id': works_int_id,
                    'query_1': query_strings[0],
                    'query_2': query_strings[1],
                    'query_3': query_strings[2],
                    'query_4': query_strings[3],
                    'query_5': query_strings[4],
                    'result_1': results[0],
                    'result_2': results[1],
                    'result_3': results[2],
                    'result_4': results[3],
                    'result_5': results[4],
                    'cited_by_count': row['citation_count']
                })

        missed_df = pl.DataFrame(missed_data)
        missed_df.write_parquet("missed_data.parquet")
        print("Saved missed works data to missed_data.parquet")

        return missed_df

    def generate_triplets(self, missed_df):
        print("Generating triplets...")
        all_queries = []
        for i in range(1, 6):
            all_queries.extend(missed_df[f'query_{i}'].to_list())

        all_query_vectors = self.model.encode(all_queries)

        query_index = faiss.IndexFlatIP(all_query_vectors.shape[1])
        query_index.add(all_query_vectors)

        triplets = []

        for i, query in enumerate(tqdm(all_queries, desc="Creating triplets")):
            anchor = query
            _, similar_indices = query_index.search(all_query_vectors[i:i + 1], 10)

            positive_idx = similar_indices[0][1]  # Second most similar (first is the query itself)
            negative_idx = similar_indices[0][-1]  # Least similar among top 10

            positive = all_queries[positive_idx]
            negative = all_queries[negative_idx]

            triplets.append((anchor, positive, negative))

        return triplets

    def fine_tune_model(self, triplets):
        print("Fine-tuning the model...")
        train_examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.TripletLoss(model=self.model)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

        new_model_path = f"{self.model_path}_finetuned"
        self.model.save(new_model_path)
        print(f"Fine-tuned model saved to {new_model_path}")

    def rebuild_vector_db(self):
        print("Rebuilding vector database...")
        all_works = pl.read_parquet(self.work_data_path)

        all_queries = []
        for row in tqdm(all_works.iter_rows(named=True), total=len(all_works), desc="Generating queries"):
            all_queries.extend(self.generate_queries(row))

        all_query_vectors = self.model.encode(all_queries, show_progress_bar=True)

        new_index = faiss.IndexFlatIP(all_query_vectors.shape[1])
        new_index.add(all_query_vectors)

        new_index_path = f"{self.index_path}_updated"
        faiss.write_index(new_index, new_index_path)
        print(f"Updated index saved to {new_index_path}")

    def run(self):
        self.load_resources()
        missed_df = self.find_missed_works()
        triplets = self.generate_triplets(missed_df)
        self.fine_tune_model(triplets)
        self.rebuild_vector_db()


# Usage
if __name__ == "__main__":
    index_path = "path/to/your/index.bin"
    model_path = "path/to/your/model"
    work_data_path = "path/to/your/work_data.parquet"

    finetuner = CloudDatasetReinforcementFinetuning(index_path, model_path, work_data_path)
    finetuner.run()