import datetime
import os
from typing import List, Dict, Any
from typing import Tuple
import faiss
import numpy as np
import polars as pl
from datasets import Dataset
from pymongo import MongoClient
from sentence_transformers import InputExample, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        I want you to provide extra steps 1-8, and the directory builder outline as well, is something we need.

        So, I will show you some code for generating the index. Its build for multiple gpu's, but you should make it run for 1 gpu.
        Since we are using 32 dimensions with matroyshka, I do not want you to quantize with product quantization. We will just use the normal embeddings
        and make them embed with HNSW and IVF.

        TODO: Make the number of epochs parameterizable. We can also choose max number of epochs but terminate if epoch n and epoch n-1 are both evaluated
            to have a recall score that is less than before.

        We need a square distance matrix for filtering out positives and negatives and anchors that are all too close to each other.

        TODO: make k for knn search a parameter in the class arguments.



    """



    def __init__(self,
                 model_path: str,
                 faiss_index_path: str,
                 triplets_path: str,
                 mongodb_url: str,
                 database_name: str,
                 output_directory: str,
                 embedding_dim: int = 32,
                 max_epochs: int = 5,
                 recall_threshold: float = 0.99):

        self.model_path = model_path
        self.faiss_index_path = faiss_index_path
        self.triplets_path = triplets_path
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.output_directory = output_directory
        self.embedding_dim = embedding_dim
        self.max_epochs = max_epochs
        self.recall_threshold = recall_threshold

        self.model = None
        self.faiss_index = None
        self.mongo_client = None
        self.db = None
        self.works_collection = None

        self.matryoshka_dims = [384, 256, 128, 64, 32]
        self.use_adaptive_layers = True

    def setup_directory_structure(self):
        """Create the directory structure for output"""
        for epoch in range(self.max_epochs):
            epoch_dir = os.path.join(self.output_directory, f"epoch_{epoch}")
            os.makedirs(os.path.join(epoch_dir, "index"), exist_ok=True)
            os.makedirs(os.path.join(epoch_dir, "model"), exist_ok=True)
            os.makedirs(os.path.join(epoch_dir, "triplets"), exist_ok=True)

    def load_resources(self):
        """Load the model, FAISS index, and connect to MongoDB"""
        print("Loading resources...")
        self.model = SentenceTransformer(self.model_path)
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        self.mongo_client = MongoClient(self.mongodb_url)
        self.db = self.mongo_client[self.database_name]
        self.works_collection = self.db['Works']

    def query_string_schema(self, work: Dict[str, Any]) -> List[str]:
        """Generate query strings based on work object fields"""
        title = work.get('display_name', '')
        primary_topic = work.get("primary_topic", {})
        field = primary_topic.get("field", {}).get("display_name", "")
        subfield = primary_topic.get("subfield", {}).get("display_name", "")
        author_names = [authorship.get('author', {}).get('display_name', '') for authorship in
                        work.get('authorships', [])]

        queries = [
            f"{title} {field}",
            f"{' '.join(author_names[:2])} {field}",
            f"{title} {subfield}",
            f"{' '.join(author_names[:2])} {subfield}",
            f"{title} {' '.join(author_names[:2])}",
        ]
        return [q.strip() for q in queries if q.strip()]

    def faiss_knn_search(self, query_vector: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Perform k-NN search using FAISS index"""
        return self.faiss_index.search(query_vector.reshape(1, -1), k)

    def find_missed_works(self) -> pl.DataFrame:
        """Find works that are not in the top k nearest neighbors for their queries"""
        missed_data = []
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
            queries = self.query_string_schema(work)
            work_int_id = work['works_int_id']

            for query in queries:
                query_vector = self.model.encode([query])[0]
                similar_indices, _ = self.faiss_knn_search(query_vector)
                if work_int_id not in similar_indices[0]:
                    missed_data.append({
                        'work_id': work['id'],
                        'works_int_id': work_int_id,
                        'query': query,
                        'cited_by_count': work['cited_by_count']
                    })

        missed_df = pl.DataFrame(missed_data)
        return missed_df

    def create_matryoshka_embeddings(self, missed_df: pl.DataFrame) -> np.ndarray:
        """Create Matryoshka embeddings for missed works"""
        all_queries = missed_df['query'].to_list()

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

        train_dataset = Dataset.from_dict({'text': all_queries})
        self.model.fit(
            train_objectives=[(train_dataset, loss)],
            epochs=1,
            warmup_steps=100,
            output_path=f"{self.output_directory}/current_epoch/matryoshka_model"
        )

        embeddings = self.model.encode(all_queries, convert_to_tensor=True)[:, :self.embedding_dim]
        return embeddings.cpu().numpy()

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a FAISS index with HNSW and IVF"""
        dimension = embeddings.shape[1]
        nlist = min(4096, int(embeddings.shape[0] / 39))  # rule of thumb for nlist
        quantizer = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors for HNSW
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        # Train and add vectors
        index.train(embeddings)
        index.add(embeddings)

        return index

    def hard_negative_triplet_mining(self, embeddings: np.ndarray, missed_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """Perform hard negative triplet mining"""
        triplets = []
        for i, anchor in enumerate(tqdm(embeddings, desc="Mining triplets")):
            _, I = self.faiss_knn_search(anchor, k=10)
            positives = I[I != i][:5]  # 5 closest, excluding self
            negatives = I[I != i][-5:]  # 5 farthest

            for pos in positives:
                for neg in negatives:
                    triplets.append({
                        'anchor': missed_df['query'][i],
                        'positive': missed_df['query'][pos],
                        'negative': missed_df['query'][neg]
                    })

        return triplets

    def mix_and_shuffle_triplets(self, new_triplets: List[Dict[str, Any]]) -> pl.DataFrame:
        """Mix new triplets with existing ones and shuffle"""
        existing_triplets = pl.read_parquet(self.triplets_path)
        all_triplets = pl.concat([existing_triplets, pl.DataFrame(new_triplets)])
        return all_triplets.sample(fraction=1.0, seed=42)  # shuffle

    def fine_tune_encoder(self, triplets_df: pl.DataFrame):
        """
        Fine-tune the encoder on the new triplet dataset using up-to-date SentenceTransformer training arguments.
        """
        current_date = datetime.datetime.now().strftime("%Y_%m_%d")
        output_path = os.path.join(self.output_directory, f'fine_tuned_model_{current_date}')
        os.makedirs(output_path, exist_ok=True)

        # Prepare the dataset
        train_dataset = Dataset.from_pandas(triplets_df)

        # Create the base loss function
        base_loss = losses.TripletLoss(model=self.model)

        # Create the Matryoshka loss
        if self.use_adaptive_layers:
            loss = losses.Matryoshka2dLoss(
                model=self.model,
                loss=base_loss,
                matryoshka_dims=self.matryoshka_dims,
                n_layers_per_step=1,
                n_dims_per_step=1
            )
            print("Using Matryoshka2dLoss with adaptive layers")
        else:
            loss = losses.MatryoshkaLoss(
                model=self.model,
                loss=base_loss,
                matryoshka_dims=self.matryoshka_dims
            )
            print("Using MatryoshkaLoss without adaptive layers")

        # Define training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=output_path,
            num_train_epochs=3,  # You can adjust this
            per_device_train_batch_size=16,
            learning_rate=2e-5,  # You can adjust this
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_dir=os.path.join(output_path, "logs"),
        )

        # Create the trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss
        )

        # Start training
        trainer.train()

        print(f"Fine-tuning completed. Model saved to {output_path}")

        # Update the model path
        self.model_path = output_path
        # Load the fine-tuned model
        self.model = SentenceTransformer(self.model_path)
    def create_embeddings_for_database(self) -> np.ndarray:
        """Create embeddings for all documents in the database"""
        all_works = self.works_collection.find({}, {'full_string': 1})
        all_strings = [work['full_string'] for work in all_works]
        return self.model.encode(all_strings, convert_to_tensor=True)[:, :self.embedding_dim].cpu().numpy()

    def run_epoch(self, epoch: int):
        """Run a single epoch of the reinforcement fine-tuning process"""
        print(f"Starting epoch {epoch}")

        # Step 1-2: Find missed works
        missed_df = self.find_missed_works()
        missed_df.write_parquet(f"{self.output_directory}/epoch_{epoch}/missed_works.parquet")

        # Step 3: Create Matryoshka embeddings
        embeddings = self.create_matryoshka_embeddings(missed_df)

        # Step 4: Index embeddings and perform hard negative triplet mining
        temp_index = self.create_faiss_index(embeddings)
        new_triplets = self.hard_negative_triplet_mining(embeddings, missed_df)

        # Step 5: Mix and shuffle triplets
        all_triplets = self.mix_and_shuffle_triplets(new_triplets)
        all_triplets.write_parquet(f"{self.output_directory}/epoch_{epoch}/triplets/all_triplets.parquet")

        # Step 6: Fine-tune the encoder
        self.fine_tune_encoder(all_triplets)

        # Step 7: Create embeddings for all documents
        all_embeddings = self.create_embeddings_for_database()

        # Step 8: Index new embeddings
        new_index = self.create_faiss_index(all_embeddings)
        faiss.write_index(new_index, f"{self.output_directory}/epoch_{epoch}/index/faiss_index.bin")

        # Update class attributes for next epoch
        self.model = SentenceTransformer(f"{self.output_directory}/epoch_{epoch}/model")
        self.faiss_index = new_index
        self.faiss_index_path = f"{self.output_directory}/epoch_{epoch}/index/faiss_index.bin"
        self.triplets_path = f"{self.output_directory}/epoch_{epoch}/triplets/all_triplets.parquet"

    def calculate_recall(self) -> float:
        """Calculate recall@20 for the current epoch"""
        # Implementation needed
        pass

    def run(self):
        """Run the entire reinforcement fine-tuning process"""
        self.setup_directory_structure()
        self.load_resources()

        for epoch in range(self.max_epochs):
            self.run_epoch(epoch)
            recall = self.calculate_recall()
            print(f"Epoch {epoch} completed. Recall@20: {recall:.4f}")

            if recall > self.recall_threshold:
                print(f"Recall threshold reached. Stopping training.")
                break

        print("Reinforcement fine-tuning completed.")


if __name__ == "__main__":
    # Example usage
    finetuner = CloudDatasetReinforcementFinetuning(
        model_path="path/to/initial/model",
        faiss_index_path="path/to/initial/faiss/index",
        triplets_path="path/to/initial/triplets.parquet",
        mongodb_url="mongodb://localhost:27017",
        database_name="OpenAlex",
        output_directory="path/to/output/directory",
        embedding_dim=32,
        max_epochs=10,
        recall_threshold=0.95
    )
    finetuner.run()