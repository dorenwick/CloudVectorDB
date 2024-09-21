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

class TrainingSentenceEncoder:
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
                 matryoshka_dims=[384, 256, 128, 64, 32],):

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

        # Create the Matryoshka2dLoss
        loss = losses.Matryoshka2dLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=self.matryoshka_dims,
            n_layers_per_step=1,
            n_dims_per_step=1
        )

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

    def load_matryoshka_model(self, model_path, num_layers=None, embedding_dim=None):
        """
        Load a trained Matryoshka model with specified number of layers and embedding dimension.

        :param model_path: Path to the trained Matryoshka model
        :param num_layers: Number of layers to use (None for all layers)
        :param embedding_dim: Embedding dimension to use (None for full dimension)
        :return: Loaded SentenceTransformer model
        """
        model = SentenceTransformer(model_path)

        if num_layers is not None:
            # Adjust the number of layers
            model.auto_model.encoder.layer = model.auto_model.encoder.layer[:num_layers]

        if embedding_dim is not None:
            # Adjust the embedding dimension
            model.dimension = embedding_dim

        return model

    def encode_sentences(self, model, sentences):
        """
        Encode a list of sentences using the loaded model.

        :param model: Loaded SentenceTransformer model
        :param sentences: List of sentences to encode
        :return: Encoded embeddings
        """
        return model.encode(sentences)

    def test_matryoshka_model(self, model_path, test_sentences):
        """
        Test the Matryoshka model with different configurations.

        :param model_path: Path to the trained Matryoshka model
        :param test_sentences: List of sentences to test
        """
        print("Testing Matryoshka model with different configurations:")

        # Test configurations
        configs = [
            (None, None, "Full model"),
            (6, None, "6 layers, full dimension"),
            (None, 256, "All layers, 256-dim"),
            (6, 256, "6 layers, 256-dim"),
            (3, 128, "3 layers, 128-dim")
        ]

        for num_layers, embedding_dim, config_name in configs:
            print(f"\nConfiguration: {config_name}")
            model = self.load_matryoshka_model(model_path, num_layers, embedding_dim)

            start_time = time.time()
            embeddings = self.encode_sentences(model, test_sentences)
            end_time = time.time()

            print(f"Embedding shape: {embeddings.shape}")
            print(f"Encoding time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    model_path = r"E:\HugeDatasetBackup\cloud_models\best_model"
    guide_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model"
    output_directory = r"E:\HugeDatasetBackup\cloud_models\matryoshka_model"
    datasets_directory = r"E:\HugeDatasetBackup\cloud_datasets"
    dataset_file = r"E:\HugeDatasetBackup\cloud_datasets\triplets_test.parquet"

    # For MultipleNegativesRankingLoss
    encoder_mnrl = TrainingSentenceEncoder(
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
        matryoshka_dims=[384, 256, 128, 64, 32],
    )

    # For GISTEmbedLoss
    encoder_gist = TrainingSentenceEncoder(
        model_path=model_path,
        output_directory=output_directory,
        datasets_directory=datasets_directory,
        guide_model_path=guide_model_path,
        batch_size=32,
        mini_batch_size=32,
        cache_batch_size=32,
        evaluator="TripletEvaluator",
        loss_function="GISTEmbedLoss",
        dataset_size=20_000,
        matryoshka_dims=[384, 256, 128, 64, 32],
    )

    # Clear CUDA cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # Test sentences
    test_sentences = [
        "This is a test sentence.",
        "Another example for encoding.",
        "Let's see how the Matryoshka model performs."
    ]

    encoder_mnrl.test_matryoshka_model(r"E:\HugeDatasetBackup\cloud_models\matryoshka_model\matryoshka_model_2024_09_21\checkpoint-1200", test_sentences)

    # Fine-tune the Matryoshka model
    # # Fine-tune the Matryoshka model with MultipleNegativesRankingLoss
    # encoder_mnrl.fine_tune_matryoshka(dataset_file, epochs=1, learning_rate=1e-5, weight_decay=0.00)
    #
    # # Fine-tune the Matryoshka model with GISTEmbedLoss
    # encoder_gist.fine_tune_matryoshka(dataset_file, epochs=1, learning_rate=1e-5, weight_decay=0.00)