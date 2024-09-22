import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer, losses
from tqdm import tqdm
from pymongo import MongoClient
import torch
from datasets import Dataset

class CloudDatasetReinforcementFinetuning:

    """
    1: The goal of this class is to create a training system for building datasets that will fine-tune the perfect model.

    TODO: To accomplish this, we will do the following. We will take an input:
        model
        model_directory
        faiss_index
        faiss_index_directory
        triplets.parquet
                triplets_directory
        database (such as mongodb), which stores the document string data. all database details here as an input.
        model_output_directory
        faiss_index_output_directory
        triplets_output_directory

        Provide a schema of query string types where query strings are built from work object fields in our mongodb database.
        The schema (and method) contains instructions on how to build particular query string types from the fields of the work object.

        1 We use faiss knn search through our index to find which works are found in the top k nearest neighbors, and then
        2 Create a dataframe and parquet file of all the works that were not found in the top k nearest neighbors, with query string types and reuslts from mongodb
         or our structured database as well as all the query types.
        3 Then, we take all of the query strings that didn't result in the search result, and we take the full_string of the work object,
        and encode them into encodings (using matryoshka, with adaptive layers).
        4 Then, we index them, and do hard negative triplet mining to create triplets.
        5 Then, we mix the triplets into our current triplet dataset, shuffle the new dataset of triplets we have made.
        6 Then, we fine-tune our previous encoder on the new dataset of triplets.
        7 Then, we use the fine-tuned encoder to create embeddings for each document in our database (mongodb, on document strings).
        8 Then, we index the new embeddings into a new index via faiss.

        At this point, we have completed one epoch and are ready to begin steps 1-8 again.
        The goal here is to construct a system for repeating epochs 1 through 8.
        A stopping point should be when the recall@20 scores aren't changing much between epochs.

        So, to build this class, we note some specific details:
            faiss index parameters need to be given to us, and by default will be calculated for us
            sentence encoder parameters need to be given to us (default is 32 dimension)

        So, we will show you our faiss index builder code.
            we will show you our sentence transformer builder code.
            we will show you our database code.

        More importantly, is the fact that we need to have a directory schema setup so that we create a new directory
        for each epoch, we we have a epoch_# output directory that contains output directories for index, triplets, and model.
        Since this will use a ton of space, we may want a system set up to keep only the best index and best model, and best dataset.
        That will be something to account for.
        But lets not make that the default system. For now the default will be to save everything to disk for each epoch.

        Currently, our class has a system built for building a missed_data.parquet file and for training a sentence encoder.
        I want you to provide extra steps 1-8, and the directory builder outline as well.







    """


    def __init__(self, index_path, model_path, mongodb_url, database_name, output_directory):
        self.index_path = index_path
        self.model_path = model_path
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.output_directory = output_directory
        self.index = None
        self.model = None
        self.mongo_client = None
        self.db = None
        self.works_collection = None
        self.matryoshka_dims = [384, 256, 128, 64, 32]
        self.use_adaptive_layers = True

    def load_resources(self):
        print("Loading index...")
        self.index = faiss.read_index(self.index_path)

        print("Loading model...")
        self.model = SentenceTransformer(self.model_path)

        print("Connecting to MongoDB...")
        self.mongo_client = MongoClient(self.mongodb_url)
        self.db = self.mongo_client[self.database_name]
        self.works_collection = self.db['Works']


    def generate_queries(self, work):
        title = work.get('display_name', '')
        primary_topic = work.get("primary_topic", {})


        if primary_topic:
            field = primary_topic.get("field", {}).get("display_name", "")
            subfield = primary_topic.get("subfield", {}).get("display_name", "")
        else:
            field = ""
            subfield = ""
        author_names = [authorship.get('author', {}).get('display_name', '') for authorship in work.get('authorships', [])]


        if title:
            title_words = title.split()
        else:
            title_words = "science"

        trigram = " ".join(title_words[:min(3, len(title_words))])

        queries = [
            f"{title} {field}",
            f"{' '.join(author_names[:2])} {field}",
        ]

        return queries

    def search_similar_works(self, query_vector, k=20):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices[0], distances[0]

    def find_missed_works(self):
        """
        We should only count a miss if the query actually has the data, so for example if it doesn't have any
        authors, then we will automatically count the query:             f"{' '.join(author_names[:2])} {field}",
        as a hit. (and make it 1 instead of 0).
        Similar logic should be used for titles.
        We ought to make a schema for specifying query types (title_query, author_query, both_query, topic_query, ect).


        :return:
        """

        missed_data = []
        good_results_count = 0
        bad_results_count = 0

        projection = {
            'id': 1,
            'works_int_id': 1,
            'display_name': 1,
            'primary_topic': 1,
            'authorships': 1,
            'cited_by_count': 1
        }

        works_cursor = self.works_collection.find(
            {'cited_by_count': {'$gte': 25}},
            projection
        ).limit(10_000)

        for work in tqdm(works_cursor, total=10_000, desc="Processing works"):
            work_id = work['id']
            works_int_id = work.get('works_int_id')
            if bad_results_count % 100 == 0:
                print("works_int_id: ", works_int_id)
            queries = self.generate_queries(work)

            results = []
            query_strings = []

            for query in queries:
                query_vector = self.model.encode([query])[0]
                similar_indices, _ = self.search_similar_works(query_vector)
                retrieved_work_int_ids = [int(idx) + 1 for idx in similar_indices]
                found = works_int_id in retrieved_work_int_ids
                results.append(int(found))
                query_strings.append(query)

            if 0 in results:
                missed_data.append({
                    'work_id': work_id,
                    'works_int_id': works_int_id,
                    'query_1': query_strings[0],
                    'query_2': query_strings[1],
                    'result_1': results[0],
                    'result_2': results[1],
                    'cited_by_count': work['cited_by_count']
                })
                bad_results_count += 1
            else:
                good_results_count += 1

        missed_df = pl.DataFrame(missed_data)
        missed_df.write_parquet("missed_data.parquet")
        print("Saved missed works data to missed_data.parquet")

        print(f"Good results (found in all queries): {good_results_count}")
        print(f"Bad results (missed in at least one query): {bad_results_count}")
        print(f"Total works processed: {good_results_count + bad_results_count}")
        print(f"Percentage of good results: {good_results_count / (good_results_count + bad_results_count) * 100:.2f}%")

        return missed_df

    def create_matryoshka_embeddings(self, missed_df):
        print("Creating Matryoshka embeddings...")

        # Combine all query strings into a single list
        all_queries = missed_df['query_1'].tolist() + missed_df['query_2'].tolist()

        # Create Matryoshka loss
        base_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        if self.use_adaptive_layers:
            loss = losses.Matryoshka2dLoss(
                model=self.model,
                loss=base_loss,
                matryoshka_dims=self.matryoshka_dims,
                n_layers_per_step=1,
                n_dims_per_step=1
            )
        else:
            loss = losses.MatryoshkaLoss(
                model=self.model,
                loss=base_loss,
                matryoshka_dims=self.matryoshka_dims
            )

        # Fine-tune the model with Matryoshka loss
        train_dataset = Dataset.from_dict({'text': all_queries})
        self.model.fit(
            train_objectives=[(train_dataset, loss)],
            epochs=1,
            warmup_steps=100,
            output_path=f"{self.output_directory}/matryoshka_model"
        )

        # Generate 32-dimensional embeddings
        embeddings = self.model.encode(all_queries, convert_to_tensor=True)[:, :32]
        return embeddings.cpu().numpy()

    def create_vectordb(self, embeddings):
        print("Creating vector database...")
        dimension = embeddings.shape[1]  # Should be 32
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def run(self):
        self.load_resources()
        missed_df = self.find_missed_works()

        # Clear CUDA cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        embeddings = self.create_matryoshka_embeddings(missed_df)
        vectordb = self.create_vectordb(embeddings)

        # Save the new index
        faiss.write_index(vectordb, f"{self.output_directory}/matryoshka_index.bin")
        print(f"Saved new Matryoshka index to {self.output_directory}/matryoshka_index.bin")


# Usage
if __name__ == "__main__":
    index_path = r"E:\HugeDatasetBackup\works_index_updated.bin"
    mongodb_url = 'mongodb://localhost:27017'
    database_name = 'OpenAlex'
    model_path = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models\best_model"
    output_directory = r"E:\HugeDatasetBackup\cloud_models\matryoshka_model"

    finetuner = CloudDatasetReinforcementFinetuning(index_path, model_path, mongodb_url, database_name, output_directory)
    finetuner.run()