import datetime
import os
import time

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper

class CloudTrainingMatryoshka():

    """
    https://huggingface.co/blog/matryoshka#why-would-you-use-%F0%9F%AA%86-matryoshka-embedding-models

    Keep in mind that although processing smaller embeddings for downstream tasks (retrieval, clustering, etc.) will be faster,
    getting the smaller embeddings from the model is just as fast as getting the larger ones.

    We shall however, use these for training indexes.

    TODO: We do not want to use adaptive layers for training, unless it meets our requirements and approval.
        The reason is that adaptive layers do not do anything in a vector database-there is no need for them there.
        They are only useful in the re-ranking process.

    """

    def __init__(self,
                 model_path,
                 output_directory,
                 datasets_directory,
                 guide_model_path,
                 batch_size=32,
                 mini_batch_size=32,
                 cache_batch_size=32,
                 evaluator="TripletEvaluator",
                 loss_function="MultipleNegativesRankingLoss",
                 dataset_size=100_000,
                 matryoshka_dims=[384, 24],
                 use_adaptive_layers=False):  # New parameter

        self.model_path = model_path
        self.output_directory = output_directory
        self.datasets_directory = datasets_directory
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.cache_batch_size = cache_batch_size
        self.evaluator = evaluator
        self.loss_function = loss_function
        self.dataset_size = dataset_size
        self.matryoshka_dims = matryoshka_dims
        self.guide_model_path = guide_model_path
        self.use_adaptive_layers = use_adaptive_layers  # New attribute


    def fine_tune_matryoshka(self, dataset_file, epochs=1, learning_rate=1e-5, weight_decay=0.00):
        current_date = datetime.datetime.now().strftime("%Y_%m_%d")
        output_path = os.path.join(self.output_directory, f'matryoshka_model_{current_date}')
        os.makedirs(output_path, exist_ok=True)

        # Load the dataset
        df = pd.read_parquet(dataset_file)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df[:self.dataset_size]

        # Initialize the SentenceTransformer model
        model = SentenceTransformer(self.model_path)

        # Create examples for training
        train_dataset = self.create_examples_multiple_negatives(df)

        # Create the base loss function
        base_loss = self.get_loss_function(self.loss_function, model)

        # Create the Matryoshka loss based on the use_adaptive_layers parameter
        if self.use_adaptive_layers:
            loss = losses.Matryoshka2dLoss(
                model=model,
                loss=base_loss,
                matryoshka_dims=self.matryoshka_dims,
                n_layers_per_step=1,
                n_dims_per_step=1
            )
            print("Using Matryoshka2dLoss with adaptive layers")
        else:
            loss = losses.MatryoshkaLoss(
                model=model,
                loss=base_loss,
                matryoshka_dims=self.matryoshka_dims
            )
            print("Using MatryoshkaLoss without adaptive layers")

        # Create an evaluator
        evaluator = self.create_evaluator(train_dataset.select(range(500)), self.evaluator)

        # Define training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_dir=os.path.join(output_path, "logs"),
            save_total_limit=2,
        )

        # Create the trainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            evaluator=evaluator,
            loss=loss
        )

        # Start training
        trainer.train()

        print(f"Matryoshka model training completed. Model saved to {output_path}")
        return output_path


    @measure_time
    def create_examples_multiple_negatives(self, df):
        anchors = df['anchor_string'].tolist()
        positives = df['positive_string'].tolist()
        negatives = df['negative_string'].tolist()

        # Create the dataset directly without creating InputExample objects
        dataset = Dataset.from_dict({
            'anchor': anchors,
            'positive': positives,
            'negative': negatives,
        })

        print(f"Created dataset with {len(dataset)} examples for Multiple Negatives")
        return dataset

    def get_loss_function(self, loss_function_name, model):
        if loss_function_name == "MultipleNegativesRankingLoss":
            return losses.MultipleNegativesRankingLoss(model=model)
        elif loss_function_name == "GISTEmbedLoss":
            if self.guide_model_path is None:
                raise ValueError("guide_model_path must be specified for GISTEmbedLoss")
            guide_model = SentenceTransformer(self.guide_model_path)
            return losses.GISTEmbedLoss(model=model, guide=guide_model)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def create_evaluator(self, test_dataset, evaluator_name):
        if evaluator_name == "TripletEvaluator":
            return TripletEvaluator(
                test_dataset['anchor'],
                test_dataset['positive'],
                test_dataset['negative']
            )
        else:
            raise ValueError(f"Unsupported evaluator: {evaluator_name}")

    def load_matryoshka_model(self, model_path, num_layers=None):
        model = SentenceTransformer(model_path)

        if num_layers is not None:
            if hasattr(model, 'auto_model'):
                transformer_model = model.auto_model
            elif hasattr(model[0], 'auto_model'):
                transformer_model = model[0].auto_model
            else:
                raise AttributeError("Cannot find the underlying transformer model")

            if hasattr(transformer_model, 'encoder'):
                transformer_model.encoder.layer = transformer_model.encoder.layer[:num_layers]
            elif hasattr(transformer_model, 'layers'):
                transformer_model.layers = transformer_model.layers[:num_layers]
            else:
                raise AttributeError("Cannot find the layers attribute in the transformer model")

        return model

    def encode_sentences(self, model, sentences, embedding_dim=None):
        embeddings = model.encode(sentences, convert_to_tensor=True)
        if embedding_dim is not None:
            embeddings = embeddings[:, :embedding_dim]
        return embeddings

    def test_matryoshka_model(self, model_path, test_sentences_file):
        print("Testing Matryoshka model with different configurations:")

        # Load test sentences
        test_sentences = pd.read_parquet(test_sentences_file)['sentence'].tolist()

        configs = [
            # (None, None, "Full model"),
            # (3, None, "3 layers, full dim"),
            # (3, 192, "3 layers, 192-dim"),
            # (3, 96, "3 layers, 96-dim"),
            # (3, 48, "3 layers, 48-dim"),
            (3, 24, "3 layers, 24-dim"),
        ]

        for num_layers, embedding_dim, config_name in configs:
            print(f"\nConfiguration: {config_name}")
            model = self.load_matryoshka_model(model_path, num_layers)

            start_time = time.time()
            embeddings = self.encode_sentences(model, test_sentences, embedding_dim)
            end_time = time.time()

            print(f"Embedding shape: {embeddings.shape}")
            print(f"Encoding time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    model_path = r"E:\HugeDatasetBackup\cloud_models\best_model"
    guide_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model"
    output_directory = r"E:\HugeDatasetBackup\cloud_models\matryoshka_model"
    datasets_directory = r"E:\HugeDatasetBackup\cloud_datasets"
    test_sentences_file = r"E:\HugeDatasetBackup\cloud_datasets\test_sentences.parquet"
    dataset_file = r"E:\HugeDatasetBackup\cloud_datasets\triplets_test.parquet"
    matryoshka_dims = [384, 192, 96, 48, 24, 12, 6]
    # matryoshka_dims = [384, 24]


    # Example usage without adaptive layers
    encoder_standard = CloudTrainingMatryoshka(
        model_path=model_path,
        output_directory=output_directory,
        datasets_directory=datasets_directory,
        guide_model_path=guide_model_path,
        batch_size=32,
        mini_batch_size=32,
        cache_batch_size=32,
        evaluator="TripletEvaluator",
        loss_function="MultipleNegativesRankingLoss",
        dataset_size=20_000,
        matryoshka_dims=matryoshka_dims,
        use_adaptive_layers=False,

    )

    # Example usage with adaptive layers
    encoder_adaptive = CloudTrainingMatryoshka(
        model_path=model_path,
        output_directory=output_directory,
        datasets_directory=datasets_directory,
        guide_model_path=guide_model_path,
        batch_size=32,
        mini_batch_size=32,
        cache_batch_size=32,
        evaluator="TripletEvaluator",
        loss_function="MultipleNegativesRankingLoss",
        dataset_size=20_000,
        matryoshka_dims=matryoshka_dims,
        use_adaptive_layers=True
    )



    # Clear CUDA cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Train the model without adaptive layers
    encoder_standard.fine_tune_matryoshka(os.path.join(datasets_directory, "triplets_test.parquet"))

    # Train the model with adaptive layers
    encoder_adaptive.fine_tune_matryoshka(os.path.join(datasets_directory, "triplets_test.parquet"))

    # Test the Matryoshka models
    encoder_adaptive.test_matryoshka_model(r"E:\HugeDatasetBackup\cloud_models\matryoshka_model\matryoshka_model_2024_09_21\checkpoint-1200", test_sentences_file)
    encoder_standard.test_matryoshka_model(r"E:\HugeDatasetBackup\cloud_models\matryoshka_model\matryoshka_model_2024_09_21\checkpoint-1200", test_sentences_file)