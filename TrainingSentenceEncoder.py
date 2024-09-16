import datetime
import gc
import math
import os
import random
import time
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from evaluate import TranslationEvaluator

from transformers import AutoModel, AutoTokenizer

from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator, BinaryClassificationEvaluator, \
    InformationRetrievalEvaluator, MSEEvaluator, ParaphraseMiningEvaluator, \
    RerankingEvaluator, SequentialEvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.losses import AdaptiveLayerLoss, CachedGISTEmbedLoss
from sentence_transformers.training_args import BatchSamplers

# from SearchTest.VectorSearchAccuracyTest import VectorSearchAccuracyTest

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

    """
    We want to implement a system where we have hyperparmaters that tell us how to sort the dataset input.
    Consider a hyperparmater that groups together sentences with similar character lengths, (like particular ranges).
    Say 0-50 characters, then 50-100 char, 100-200, 200+ groupings. we could split the dataset into these categories
    and then concatenate them.
    Make a parameter for that, which takes in a list of integers and splits the dataframe by:
    anchor, positive, negative all being less than len(index[n]) but more than length at index[n-1]) or 0 if n=0.
    [0, 50, 100, 200, 500, 1000, 2000] would all one such list. And any that are above 2000 go into the 2000 or less part
    of the partition.

    """


    def __init__(self,
                 model_path,
                 output_directory,
                 datasets_directory,
                 batch_size=32,
                 mini_batch_size=32,  # New parameter
                 cache_batch_size=1024,
                 evaluator="BinaryClassificationEvaluator",
                 loss_function="ContrastiveLoss",
                 base_loss_function="MultipleNegativesRankingLoss",
                 guide_model_path="Snowflake/snowflake-arctic-embed-xs",
                 gist_temperature=0.01,
                 gist_mini_batch_size=32,
                 dataset_size=500000,
                 num_knn_pairs=20_000_000,
                 num_works_collected=5_000_000,
                 mongo_url="mongodb://localhost:27017/",
                 mongo_database_name="CiteGrab",
                 mongo_works_collection_name="Works"):

        self.model_path = model_path
        self.output_directory = output_directory
        self.datasets_directory = datasets_directory

        self.works_all_collections_dir = os.path.join(self.datasets_directory, "works_all_collections")
        os.makedirs(self.works_all_collections_dir, exist_ok=True)

        # File paths as class attributes
        self.works_all_collected_file = os.path.join(datasets_directory, "works_all_collected.parquet")
        self.works_common_authors_file = os.path.join(datasets_directory, "works_common_authors.parquet")
        self.works_common_titles_file = os.path.join(datasets_directory, "common_title_works.parquet")
        self.works_knn_search_file = os.path.join(datasets_directory, "works_knn_search.parquet")
        self.softer_negatives_pool_file = os.path.join(datasets_directory, "hard_negatives_pool.parquet")
        self.works_augmented_data_file = os.path.join(datasets_directory, "works_augmented_data.parquet")
        self.triplet_work_ids_only_file = os.path.join(self.datasets_directory, "triplet_work_ids_only.parquet")
        self.embeddings_file = os.path.join(datasets_directory, "embeddings_32d_with_clusters_and_density.parquet")
        self.id_mapping_works_file = os.path.join(output_directory, "id_mapping_works.arrow")
        self.unigram_data_file = os.path.join(datasets_directory, "unigram_data.arrow")
        self.bigram_data_file = os.path.join(datasets_directory, "bigram_data.arrow")
        self.index_works_file = os.path.join(output_directory, "index_works.bin")

        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size  # New attribute
        self.cache_batch_size = cache_batch_size

        self.guide_model_path = guide_model_path
        self.gist_temperature = gist_temperature
        self.gist_mini_batch_size = gist_mini_batch_size

        self.evaluator = evaluator if evaluator else self.get_default_evaluator(loss_function)
        self.loss_function = loss_function
        self.dataset_size = dataset_size
        self.num_knn_pairs = num_knn_pairs
        self.num_works_collected = num_works_collected

        self.mongo_url = mongo_url
        self.mongo_database_name = mongo_database_name
        self.mongo_works_collection_name = mongo_works_collection_name

        self.mongo_client = None
        self.mongo_db = None
        self.mongodb_works_collection = None

        self.embeddings_output_directory = os.path.join(self.datasets_directory, 'work_embeddings')
        os.makedirs(self.embeddings_output_directory, exist_ok=True)

        self.unigram_path = os.path.join(self.datasets_directory, 'unigram_data.arrow')
        self.bigram_path = os.path.join(self.datasets_directory, 'bigram_data.arrow')

        self.model = SentenceTransformer(self.model_path)

        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.work_id_search_count = {}
        self.works_knn_search = []
        self.vector_index = None
        self.faiss_to_work_id_mapping = None
        self.works_df = None
        self.work_details = {}

        self.loss_functions = {
            "BatchAllTripletLoss": losses.BatchAllTripletLoss,
            "BatchHardSoftMarginTripletLoss": losses.BatchHardSoftMarginTripletLoss,
            "ContrastiveLoss": losses.ContrastiveLoss,
            "BatchHardTripletLoss": losses.BatchHardTripletLoss,
            "OnlineContrastiveLoss": losses.OnlineContrastiveLoss,
            "CosineSimilarityLoss": losses.CosineSimilarityLoss,
            "MultipleNegativesRankingLoss": losses.MultipleNegativesRankingLoss,
            "CoSENTLoss": losses.CoSENTLoss,
            "BatchSemiHardTripletLoss": losses.BatchSemiHardTripletLoss,
            "ContrastiveTensionLoss": losses.ContrastiveTensionLoss,
            "ContrastiveTensionLossInBatchNegatives": losses.ContrastiveTensionLossInBatchNegatives,
            "AnglELoss": losses.AnglELoss,
            "DenoisingAutoEncoderLoss": losses.DenoisingAutoEncoderLoss,
            "GISTEmbedLoss": losses.GISTEmbedLoss,
            "CachedGISTEmbedLoss": losses.CachedGISTEmbedLoss,
            "MSELoss": losses.MSELoss,
            "MarginMSELoss": losses.MarginMSELoss,
            "MatryoshkaLoss": losses.MatryoshkaLoss,
            "Matryoshka2dLoss": losses.Matryoshka2dLoss,
            "AdaptiveLayerLoss": losses.AdaptiveLayerLoss,
            "MegaBatchMarginLoss": losses.MegaBatchMarginLoss,
            "CachedMultipleNegativesRankingLoss": losses.CachedMultipleNegativesRankingLoss,
            "MultipleNegativesSymmetricRankingLoss": losses.MultipleNegativesSymmetricRankingLoss,
            "SoftmaxLoss": losses.SoftmaxLoss,
            "TripletLoss": losses.TripletLoss,
        }

        self.evaluators = {
            "TripletEvaluator": TripletEvaluator,
            "EmbeddingSimilarityEvaluator": EmbeddingSimilarityEvaluator,
            "BinaryClassificationEvaluator": BinaryClassificationEvaluator,
            "InformationRetrievalEvaluator": InformationRetrievalEvaluator,
            "MSEEvaluator": MSEEvaluator,
            "ParaphraseMiningEvaluator": ParaphraseMiningEvaluator,
            "RerankingEvaluator": RerankingEvaluator,
            "SequentialEvaluator": SequentialEvaluator,
            "TranslationEvaluator": TranslationEvaluator,
        }

    def create_model_directories(self):
        base_dir = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\models"
        domains = ['health_sciences', 'life_sciences', 'physical_sciences', 'social_sciences']
        data_types = ['string', 'full']

        model_dirs = {}
        for domain in domains:
            for data_type in data_types:
                dir_name = f"{domain}_{data_type}"
                full_path = os.path.join(base_dir, dir_name)
                os.makedirs(full_path, exist_ok=True)
                model_dirs[dir_name] = full_path

        return model_dirs

    @measure_time
    def load_and_print_data(self):
        parquet_files = [
            "works_augmented_data.parquet",
            "works_common_authors.parquet",
            "works_all_collected.parquet",
            "works_knn_search.parquet",
            "hard_negatives_pool.parquet",
        ]

        for file in parquet_files:
            file_path = os.path.join(self.datasets_directory, file)
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)

                print(f"\nFile: {file}")
                print("Schema:")
                print(df.dtypes.to_string())

                print("\nHead (50 rows):")
                print(df.head(50).to_string())

                print("\nTail (50 rows):")
                print(df.tail(50).to_string())

                # Force garbage collection
                del df
                gc.collect()
            else:
                print(f"File not found: {file}")


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        allowed_optimizers = [
            "adamw_torch",
            "adamw_torch_8bit",
            "adamw_torch_fused",
            "adamw_apex_fused",
            "adamw_anyprecision",
            "adafactor",
            "galore_adamw",
            "galore_adamw_8bit",
            "galore_adafactor"
        ]
        if value not in allowed_optimizers:
            raise ValueError(f"Optimizer must be one of {allowed_optimizers}")
        self._optimizer = value


    @measure_time
    def fine_tune_hyperparameter_testing(self, batch_sizes,
                                         epochs=1,
                                         base_learning_rate=1e-6,
                                         weight_decay=0.0,
                                         max_grad_norm=1.0,
                                         batch_start_index=0,
                                         scale=20.0,
                                         train_matryoshka=False,
                                         adaptive_layers=False,
                                         dataset_file=None,
                                         model_save_dir=None):

        previous_model_path = None
        for batch_size in batch_sizes:
            self.batch_size = batch_size  # Update the batch_size attribute
            model_path = self.fine_tune_encoder_large_batch(
                previous_model_path=previous_model_path,
                epochs=epochs,
                checkpoint_save_steps=16 * 2048 // self.batch_size,
                warmup_steps=16 * 2048 // self.batch_size,
                evaluation_steps=16 * 2048 // self.batch_size,
                scale=scale,
                train_matryoshka=train_matryoshka,
                adaptive_layers=adaptive_layers,
                batch_start_index=batch_start_index,
                base_learning_rate=base_learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                dataset_file=dataset_file,
                model_save_dir=model_save_dir,
            )
            print(f"Completed fine-tuning for batch_size={self.batch_size}")
            print(f"Model saved at: {model_path}")
            previous_model_path = model_path


    def fine_tune_encoder_large_batch(self, previous_model_path=None,
                                      epochs=1, checkpoint_save_steps=2000,
                                      warmup_steps=500, evaluation_steps=500, train_matryoshka=False,
                                      scale=20.0,
                                      adaptive_layers=False, use_sorted_data=True, batch_start_index=0,
                                      base_learning_rate=1e-6, weight_decay=0.0, max_grad_norm=1.0,
                                      dataset_file=None, model_save_dir=None):

        print(f"warmup steps: {warmup_steps}")
        current_date = datetime.datetime.now().strftime("%Y_%m_%d")

        # Define training arguments
        model_type = ""
        if train_matryoshka and adaptive_layers:
            model_type = "_matryoshka2d"
        elif train_matryoshka:
            model_type = "_matryoshka"
        elif adaptive_layers:
            model_type = "_adaptive"

        if model_save_dir:
            output_path = os.path.join(model_save_dir, f'best_model_bs{self.batch_size}{model_type}_{current_date}')
        else:
            output_path = os.path.join(self.output_directory, f'best_model_bs{self.batch_size}{model_type}_{current_date}')

        df = pd.read_parquet(dataset_file)

        # # Only shuffle if not using sorted data
        # if not use_sorted_data:
        #     df = df.sample(frac=1).reset_index(drop=True)

        df = df.sample(frac=1).reset_index(drop=True)

        # df = df[:100_000]

        self.dataset_size = len(df)

        # Calculate and print the final checkpoint path
        final_checkpoint_path = self.calculate_final_checkpoint_path(self.dataset_size, checkpoint_save_steps)
        print(f"Expected final checkpoint path: {final_checkpoint_path}")

        # Print head and tail of the DataFrame
        print("\nHead of the DataFrame (100 rows):")
        print(df.head(100).to_string())
        print("\nTail of the DataFrame (100 rows):")
        print(df.tail(100).to_string())

        print("Schema of the loaded dataset:")
        print(df.dtypes)

        gc.collect()

        # Initialize the SentenceTransformer model
        if previous_model_path and os.path.exists(previous_model_path):
            model = SentenceTransformer(previous_model_path)
            print(f"Loaded fine-tuned model from previous batch: {previous_model_path}")
        else:
            model = SentenceTransformer(self.model_path)
            print("Loaded initial model")

        effective_batch_size = self.batch_size
        if self.loss_function == "CachedMultipleNegativesRankingLoss":
            effective_batch_size = self.batch_size * 1

        print(f"batch size: {self.batch_size}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Mini batch size: {self.mini_batch_size}")

        if adaptive_layers:
            base_loss = self.get_loss_function(self.loss_function, model, scale)
            loss = losses.AdaptiveLayerLoss(model=model, loss=base_loss)
        else:
            loss = self.get_loss_function(self.loss_function, model, scale)

        test_examples, test_dataset, train_dataset = self.create_examples_for_loss_function(
            df, self.loss_function, self.dataset_size, shuffle=not use_sorted_data
        )

        # Create an evaluator
        evaluator_instance = self.create_evaluator(test_examples, self.evaluator)


        # Ensure the output and checkpoint directories exist
        os.makedirs(output_path, exist_ok=True)

        print("checkpoint_save_steps: ", checkpoint_save_steps)
        print("warmup_steps: ", warmup_steps)
        print("evaluation_steps: ", evaluation_steps)

        gc.collect()

        REFERENCE_BATCH_SIZE = 32  # Choose a reference batch size, e.g., 32

        # Calculate learning rate based on effective batch size
        learning_rate = base_learning_rate * (effective_batch_size / REFERENCE_BATCH_SIZE)

        print(f"base_learning_rate: {base_learning_rate}")
        print(f"Calculated learning rate: {learning_rate}")

        print("weight_decay: ", weight_decay)

        print("max_grad_norm: ", max_grad_norm)

        # print("scale: ", scale)

        # Calculate the total number of training steps
        total_steps = len(train_dataset) * epochs // effective_batch_size
        # num_cycles = 0.5,  # You can adjust this
        min_lr_ratio = 0.1  # You can adjust this

        # Define training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            gradient_accumulation_steps=1,
            adam_epsilon=1e-8,
            adam_beta1=0.9,
            adam_beta2=0.999,
            per_device_train_batch_size=effective_batch_size,
            per_device_eval_batch_size=effective_batch_size,
            warmup_steps=warmup_steps,
            evaluation_strategy="steps",
            eval_steps=evaluation_steps,
            save_strategy="steps",
            #log_level="info",
            #log_level_replica="warning",
            #log_on_each_node=True,
            logging_dir=os.path.join(output_path, "logs"),
            # save_safetensors=True,
            save_steps=checkpoint_save_steps,
            save_total_limit=200,
            # logging_steps=32,
            fp16=False,
            bf16=True,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            run_name=f"fine_tune_bs{effective_batch_size}{model_type}",
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # Prevent duplicate sentences in a batch
            lr_scheduler_type="linear",  # We'll override this with our custom scheduler
            optim="adamw_torch",
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            loss=loss,
            evaluator=evaluator_instance,
        )


        # Start training
        trainer.train()

        print(f"Fine-tuning completed for batch_size={self.batch_size}{model_type}. Model saved to {output_path}")
        return output_path

    def create_examples_for_loss_function(self, df, loss_function, dataset_size, shuffle=False):
        if loss_function in ["CachedGISTEmbedLoss", "GISTEmbedLoss"]:
            return self.create_examples_gist(df, dataset_size, shuffle)
        elif loss_function in ["CachedMultipleNegativesRankingLoss", "MultipleNegativesRankingLoss", "TripletLoss", "BatchSemiHardTripletLoss"]:
            return self.create_examples_multiple_negatives(df, dataset_size, shuffle)
        elif loss_function == "CoSENTLoss":
            return self.create_examples_cosent(df, dataset_size, shuffle)
        elif loss_function == "ContrastiveLoss":
            return self.create_examples_contrastive(df, dataset_size, shuffle)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

    @measure_time
    def create_examples_multiple_negatives(self, df, dataset_size, shuffle=False):
        examples = []
        for _, row in df[:dataset_size].iterrows():
            if pd.notna(row['anchor_string']) and pd.notna(row['positive_string']) and pd.notna(row['negative_string']):
                examples.append(
                    InputExample(texts=[row['anchor_string'], row['positive_string'], row['negative_string']]))

        # Only shuffle if specified
        if shuffle:
            random.shuffle(examples)

        # Split into test and train
        split = min(500, len(examples) // 10)
        test_examples = examples[:split]
        train_examples = examples[split:]

        print(
            f"Created {len(test_examples)} test examples and {len(train_examples)} train examples for Multiple Negatives")

        # Convert to Datasets
        train_dataset = Dataset.from_dict({
            'anchor': [example.texts[0] for example in train_examples],
            'positive': [example.texts[1] for example in train_examples],
            'negative': [example.texts[2] for example in train_examples],
        })

        test_dataset = Dataset.from_dict({
            'anchor': [example.texts[0] for example in test_examples],
            'positive': [example.texts[1] for example in test_examples],
            'negative': [example.texts[2] for example in test_examples],
        })

        return test_examples, test_dataset, train_dataset

    @measure_time
    def create_examples_cosent(self, df, dataset_size, shuffle=False):
        examples = []
        for _, row in df[:dataset_size].iterrows():
            if pd.notna(row['anchor_string']) and pd.notna(row['positive_string']):
                examples.append({
                    'texts': [row['anchor_string'], row['positive_string']],
                    'label': float(row['z_score_pos'])
                })
            if pd.notna(row['anchor_string']) and pd.notna(row['negative_string']):
                examples.append({
                    'texts': [row['anchor_string'], row['negative_string']],
                    'label': float(row['z_score_neg'])
                })
        if shuffle:
            random.shuffle(examples)
        else:
            pass
            # random.shuffle(examples)

        # Split into test and train
        split = min(500, len(examples) // 10)
        test_examples = examples[:split]
        train_examples = examples[split:]

        print(f"Created {len(test_examples)} test examples and {len(train_examples)} train examples for CoSENT")

        # Convert train_examples to a Dataset
        train_dataset = Dataset.from_dict({
            'sentence1': [example['texts'][0] for example in train_examples],
            'sentence2': [example['texts'][1] for example in train_examples],
            'score': [example['label'] for example in train_examples],
        })

        # Convert test_examples to a Dataset
        test_dataset = Dataset.from_dict({
            'sentence1': [example['texts'][0] for example in test_examples],
            'sentence2': [example['texts'][1] for example in test_examples],
            'score': [example['label'] for example in test_examples],
        })

        return test_examples, test_dataset, train_dataset

    @measure_time
    def create_examples_contrastive(self, df, dataset_size):
        examples = []
        for _, row in df[:dataset_size].iterrows():
            if pd.notna(row['anchor_string']) and pd.notna(row['positive_string']):
                examples.append({
                    'texts': [row['anchor_string'], row['positive_string']],
                    'label': 1
                })
            if pd.notna(row['anchor_string']) and pd.notna(row['negative_string']):
                examples.append({
                    'texts': [row['anchor_string'], row['negative_string']],
                    'label': 0
                })

        # Shuffle the examples
        random.shuffle(examples)

        # Split into test and train
        split = min(500, len(examples) // 10)
        test_examples = examples[:split]
        train_examples = examples[split:]

        print(f"Created {len(test_examples)} test examples and {len(train_examples)} train examples for Contrastive")

        # Convert train_examples to a Dataset
        train_dataset = Dataset.from_dict({
            'sentence1': [example['texts'][0] for example in train_examples],
            'sentence2': [example['texts'][1] for example in train_examples],
            'label': [example['label'] for example in train_examples],
        })

        # Convert test_examples to a Dataset
        test_dataset = Dataset.from_dict({
            'sentence1': [example['texts'][0] for example in test_examples],
            'sentence2': [example['texts'][1] for example in test_examples],
            'label': [example['label'] for example in test_examples],
        })

        return test_examples, test_dataset, train_dataset

    def create_examples_gist(self, df, dataset_size, shuffle=False):
        examples = []
        for _, row in df[:dataset_size].iterrows():
            if pd.notna(row['anchor_string']) and pd.notna(row['positive_string']) and pd.notna(row['negative_string']):
                examples.append(
                    InputExample(texts=[row['anchor_string'], row['positive_string'], row['negative_string']]))

        # if shuffle:
        #     random.shuffle(examples)

        # Split into test and train
        split = min(500, len(examples) // 10)
        test_examples = examples[:split]
        train_examples = examples[split:]

        print(f"Created {len(test_examples)} test examples and {len(train_examples)} train examples for GIST")

        # Convert train_examples to a Dataset
        train_dataset = Dataset.from_dict({
            'anchor': [example.texts[0] for example in train_examples],
            'positive': [example.texts[1] for example in train_examples],
            'negative': [example.texts[2] for example in train_examples],
        })

        # Convert test_examples to a Dataset
        test_dataset = Dataset.from_dict({
            'anchor': [example.texts[0] for example in test_examples],
            'positive': [example.texts[1] for example in test_examples],
            'negative': [example.texts[2] for example in test_examples],
        })

        return test_examples, test_dataset, train_dataset


    @measure_time
    def create_examples_multiple_negatives(self, df, dataset_size, shuffle=False):
        examples = []
        for _, row in df[:dataset_size].iterrows():
            if pd.notna(row['anchor_string']) and pd.notna(row['positive_string']) and pd.notna(
                    row['negative_string']):
                examples.append(
                    InputExample(texts=[row['anchor_string'], row['positive_string'], row['negative_string']]))

        if shuffle:
            random.shuffle(examples)

        # Split into test and train
        split = min(500, len(examples) // 10)
        test_examples = examples[:split]
        train_examples = examples[split:]

        print(
            f"Created {len(test_examples)} test examples and {len(train_examples)} train examples for Multiple Negatives")

        train_dataset = Dataset.from_dict({
            'anchor': [example.texts[0] for example in train_examples],
            'positive': [example.texts[1] for example in train_examples],
            'negative': [example.texts[2] for example in train_examples],
        })

        test_dataset = Dataset.from_dict({
            'anchor': [example.texts[0] for example in test_examples],
            'positive': [example.texts[1] for example in test_examples],
            'negative': [example.texts[2] for example in test_examples],
        })

        return test_examples, test_dataset, train_dataset

    @measure_time
    def create_examples_triplet(self, df, dataset_size, shuffle=False):
        examples = []
        for _, row in df[:dataset_size].iterrows():
            if pd.notna(row['anchor_string']) and pd.notna(row['positive_string']) and pd.notna(
                    row['negative_string']):
                examples.append(InputExample(
                    texts=[row['anchor_string'], row['positive_string'], row['negative_string']],
                    label=[0, 1, 2],
                ))

        if shuffle:
            random.shuffle(examples)

        # Split into test and train
        split = min(500, len(examples) // 10)
        test_examples = examples[:split]
        train_examples = examples[split:]

        print(
            f"Created {len(test_examples)} test examples and {len(train_examples)} train examples for Triplet Loss")

        train_dataset = Dataset.from_dict({
            'texts': [example.texts for example in train_examples],
            'labels': [example.label for example in train_examples],
        })

        test_dataset = Dataset.from_dict({
            'texts': [example.texts for example in test_examples],
            'labels': [example.label for example in test_examples],
        })

        return test_examples, test_dataset, train_dataset
    # loss_class: Type[BatchAllTripletLoss | BatchHardSoftMarginTripletLoss | ContrastiveLoss | BatchHardTripletLoss | OnlineContrastiveLoss | CosineSimilarityLoss | MultipleNegativesRankingLoss | CoSENTLoss | BatchSemiHardTripletLoss | ContrastiveTensionLoss] = self.loss_functions[loss_function_name]
    def get_loss_function(self, loss_function_name, model, scale):
        if loss_function_name in self.loss_functions:
            loss_class = self.loss_functions[loss_function_name]
            if loss_function_name == "AdaptiveLayerLoss":
                base_loss = self.loss_functions["MultipleNegativesRankingLoss"](model=model)
                return AdaptiveLayerLoss(model=model, loss=base_loss)
            elif loss_function_name == "CachedMultipleNegativesRankingLoss":
                # base_loss = self.loss_functions["MultipleNegativesRankingLoss"](model=model)
                return loss_class(model=model, scale=scale)
            elif loss_function_name in ["CachedGISTEmbedLoss", "GISTEmbedLoss"]:
                guide_model = SentenceTransformer(self.guide_model_path)
                return CachedGISTEmbedLoss(model=model,
                                           guide=guide_model,
                                           temperature=self.gist_temperature,
                                           mini_batch_size=self.gist_mini_batch_size)
            elif loss_function_name in ["MultipleNegativesRankingLoss", "CoSENTLoss"]:
                return loss_class(model=model, scale=scale)
            elif loss_function_name in ["BatchAllTripletLoss", "BatchHardTripletLoss", "BatchSemiHardTripletLoss", "TripletLoss"]:
                # triplet_margin = 1.0
                return loss_class(model=model)
            elif loss_function_name == "BatchHardSoftMarginTripletLoss":
                return loss_class(model=model)
            elif loss_function_name in ["ContrastiveLoss", "OnlineContrastiveLoss"]:
                return loss_class(model=model)
            elif loss_function_name == "CosineSimilarityLoss":
                return loss_class(model=model)
            elif loss_function_name == "ContrastiveTensionLoss":
                return loss_class(model=model)
            else:
                return loss_class(model=model)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def truncate_model_layers(self, model, new_num_layers):
        model[0].auto_model.encoder.layer = model[0].auto_model.encoder.layer[:new_num_layers]
        return model

    def create_evaluator(self, test_examples, evaluator_name):
        if evaluator_name not in self.evaluators:
            raise ValueError(f"Unsupported evaluator: {evaluator_name}")

        evaluator_class = self.evaluators[evaluator_name]

        if evaluator_name == "TripletEvaluator":
            anchors = [example.texts[0] for example in test_examples]
            positives = [example.texts[1] for example in test_examples]
            negatives = [example.texts[2] for example in test_examples]
            return evaluator_class(anchors, positives, negatives)
        elif evaluator_name == "EmbeddingSimilarityEvaluator":
            sentences1 = [example['texts'][0] for example in test_examples]
            sentences2 = [example['texts'][1] for example in test_examples]
            scores = [float(example['label']) for example in test_examples]
            return evaluator_class(sentences1, sentences2, scores)
        elif evaluator_name == "BinaryClassificationEvaluator":
            sentences1 = [example['texts'][0] for example in test_examples]
            sentences2 = [example['texts'][1] for example in test_examples]
            labels = [float(example['label']) for example in test_examples]
            return evaluator_class(sentences1, sentences2, labels)
        else:
            raise ValueError(f"Unsupported evaluator: {evaluator_name}")

    def get_default_evaluator(self, loss_function_name):
        evaluator_map = {
            "MultipleNegativesRankingLoss": "TripletEvaluator",
            "CachedMultipleNegativesRankingLoss": "TripletEvaluator",
            "CoSENTLoss": "EmbeddingSimilarityEvaluator",
            "ContrastiveLoss": "BinaryClassificationEvaluator",
            "TripletLoss": "TripletEvaluator",
            "BatchAllTripletLoss": "TripletEvaluator",
            "BatchHardTripletLoss": "TripletEvaluator",
            "BatchSemiHardTripletLoss": "TripletEvaluator",
            "BatchHardSoftMarginTripletLoss": "TripletEvaluator",
        }
        return evaluator_map.get(loss_function_name, "TripletEvaluator")  # Default to TripletEvaluator

    def calculate_final_checkpoint_path(self, dataset_size, checkpoint_save_steps):
        total_steps = math.ceil(dataset_size / self.batch_size)
        num_checkpoints = math.floor(total_steps / checkpoint_save_steps)
        final_checkpoint_number = num_checkpoints * checkpoint_save_steps

        current_date = datetime.datetime.now().strftime("%Y_%m_%d")
        model_type = ""
        if hasattr(self, 'train_matryoshka') and self.train_matryoshka:
            model_type += "_matryoshka"
        if hasattr(self, 'adaptive_layers') and self.adaptive_layers:
            model_type += "_adaptive"

        output_path = os.path.join(self.output_directory, f'best_model_bs{self.batch_size}{model_type}_{current_date}')
        final_checkpoint_path = os.path.join(output_path, f"checkpoint-{final_checkpoint_number}")

        return final_checkpoint_path

    def train_all_datasets(self,):

        def print_gpu_memory():
            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("CUDA is not available. Running on CPU.")

        # Print initial GPU memory state
        print("Initial GPU memory state:")
        print_gpu_memory()

        base_dir = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER"
        datasets_dir = os.path.join(base_dir, "dataset_triplets_new", "split_triplets")
        models_dir = os.path.join(base_dir, "models")

        model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model"
        guide_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model"

        gist_temperature = 0.01
        batch_sizes = [12]
        base_learning_rate = 1e-7
        weight_decay = 0.0
        max_grad_norm = 1.0
        scale = 20.0
        epochs = 1
        batch_start_index = 0

        # Create an instance of TrainingSentenceEncoder to use create_model_directories method
        temp_encoder = TrainingSentenceEncoder(
            model_path=model_path,
            output_directory=models_dir,
            datasets_directory=datasets_dir,
            batch_size=12,
            mini_batch_size=12,
            cache_batch_size=12,
            evaluator="TripletEvaluator",
            loss_function="MultipleNegativesRankingLoss",
            base_loss_function="MultipleNegativesRankingLoss",
            guide_model_path=guide_model_path,
            gist_temperature=gist_temperature,
            gist_mini_batch_size=12,
            dataset_size=16_000_000,
            num_knn_pairs=5_000_000,
            num_works_collected=5_000_000,
            mongo_url="mongodb://localhost:27017/",
            mongo_database_name="CiteGrab",
            mongo_works_collection_name="Works"
        )

        model_dirs = temp_encoder.create_model_directories()

        # Delete the temp_encoder object
        del temp_encoder

        # Call the garbage collector
        gc.collect()

        print_gpu_memory()

        # Clear CUDA cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_gpu_memory()

        # domains = ['physical_sciences', 'social_sciences', 'health_sciences', 'life_sciences']
        domains = ['social_sciences']
        # domains = ['health_sciences', 'life_sciences', 'physical_sciences', 'social_sciences']
        data_types = ['full']

        for domain in domains:
            for data_type in data_types:
                dataset_file = os.path.join(datasets_dir, f"triplets_{domain}_{data_type}.parquet")
                model_save_dir = model_dirs[f"{domain}_{data_type}"]

                print(f"Training on {domain} {data_type} dataset")

                encoder = TrainingSentenceEncoder(
                    model_path=model_path,
                    output_directory=model_save_dir,
                    datasets_directory=datasets_dir,
                    batch_size=12,
                    mini_batch_size=12,
                    cache_batch_size=12,
                    evaluator="TripletEvaluator",
                    loss_function="MultipleNegativesRankingLoss",
                    base_loss_function="MultipleNegativesRankingLoss",
                    guide_model_path=guide_model_path,
                    gist_temperature=gist_temperature,
                    gist_mini_batch_size=12,
                    dataset_size=10_000_000,
                    num_knn_pairs=5_000_000,
                    num_works_collected=5_000_000,
                    mongo_url="mongodb://localhost:27017/",
                    mongo_database_name="CiteGrab",
                    mongo_works_collection_name="Works"
                )
                print_gpu_memory()

                encoder.fine_tune_hyperparameter_testing(
                    batch_sizes=batch_sizes,
                    epochs=epochs,
                    base_learning_rate=base_learning_rate,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    batch_start_index=batch_start_index,
                    scale=scale,
                    train_matryoshka=True,
                    adaptive_layers=False,
                    dataset_file=dataset_file,
                    model_save_dir=model_save_dir
                )
                print_gpu_memory()

                print(f"Finished training on {domain} {data_type} dataset")
                print("=" * 50)


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def save_pretrained(self, output_path):
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))




if __name__ == "__main__":

    # print(f"Current Transformers version: {transformers.__version__}")
    # we gotta go back to transformers version 4.39 afterwards, to get span marker to work.

    # ## Package Plan ##
    #
    #   environment location: C:\Users\doren\.conda\envs\Cite_Grabber
    #
    #   added / updated specs:
    #     - transformers
    #
    #
    # The following packages will be downloaded:
    #
    #     package                    |            build
    #     ---------------------------|-----------------
    #     sacremoses-master          |             py_0         404 KB  huggingface
    #     tokenizers-0.11.4          |  py310he5181cf_1         2.4 MB
    #     transformers-4.33.3        |             py_0         3.8 MB  huggingface
    #     ------------------------------------------------------------
    #                                            Total:         6.6 MB
    #
    # The following NEW packages will be INSTALLED:
    #
    #   sacremoses         huggingface/noarch::sacremoses-master-py_0
    #   tokenizers         pkgs/main/win-64::tokenizers-0.11.4-py310he5181cf_1
    #   transformers       huggingface/noarch::transformers-4.33.3-py_0

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # model_path = "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\models\\models--sentence-transformers--all-MiniLM-L6-v2\\snapshots\\8b3219a92973c328a8e22fadcfa821b5dc75636a"
    # model_path = r"C:\Users\doren\.cache\huggingface\hub\models--Snowflake--snowflake-arctic-embed-xs\snapshots\86a07656cc240af5c7fd07bac2f05baaafd60401"

    model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model"

    # guide_model = SentenceTransformer(model_path)

    # model_path = "Snowflake/snowflake-arctic-embed-xs"
    # guide_model_path = "Snowflake/snowflake-arctic-embed-xs"
    # model_path = r"C:\Users\doren\.cache\huggingface\hub\models--Snowflake--snowflake-arctic-embed-m\snapshots\2ca412ec9505022eebd7d10286fbbad4b779f6e0"
    # model_path =r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs8_2024_08_14\checkpoint-55296"
    guide_model_path = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs32_2024_08_12\checkpoint-124985"
    # 128

    output_directory = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models"
    datasets_directory = r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\datasets"
    # CachedMultipleNegativesRankingLoss
    # CachedGISTEmbedLoss
    # GISTEmbedLoss
    # MultipleNegativesRankingLoss

    # if model is overfitting, lower temperature.
    # if model is underfitting, upper temperature.

    gist_temperature = 0.01
    # gist_temperature = 0.001
    batch_start_index = 0
    batch_sizes = [32]
    base_learning_rate = 1e-7
    weight_decay = 0.0
    max_grad_norm = 1.0
    scale = 20.0
    epochs = 2

    #   - String data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_physical_sciences_string.parquet
    #   - Full data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_physical_sciences_full.parquet
    #   - String data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_social_sciences_string.parquet
    #   - Full data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_social_sciences_full.parquet
    #   - String data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_life_sciences_string.parquet
    #   - Full data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_life_sciences_full.parquet
    #   - String data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_health_sciences_string.parquet
    #   - Full data: E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_health_sciences_full.parquet


    # TODO: train guide model on the large snowflake model size.


    encoder = TrainingSentenceEncoder(
        model_path=model_path,
        output_directory=output_directory,
        datasets_directory=datasets_directory,
        batch_size=32,
        mini_batch_size=32,  # This will be used as the mini_batch_size for CachedMultipleNegativesRankingLoss
        cache_batch_size=32,
        evaluator="TripletEvaluator",
        loss_function="MultipleNegativesRankingLoss",
        base_loss_function="MultipleNegativesRankingLoss",
        guide_model_path=guide_model_path,
        gist_temperature=gist_temperature,
        gist_mini_batch_size=32,
        dataset_size=16_000_000,
        num_knn_pairs=5_000_000,
        num_works_collected=5_000_000,
        mongo_url="mongodb://localhost:27017/",
        mongo_database_name="CiteGrab",
        mongo_works_collection_name="Works"
    )

    # Clear CUDA cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_physical_sciences_full.parquet"
    # dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_life_sciences_full.parquet"
    # dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_health_sciences_full.parquet"
    # dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_social_sciences_full.parquet"

    encoder.train_all_datasets()

    dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_social_sciences_string.parquet"

    encoder.fine_tune_hyperparameter_testing(
        batch_sizes,
        epochs=epochs,
        base_learning_rate=base_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        batch_start_index=batch_start_index,
        scale=scale,
        train_matryoshka=False,
        adaptive_layers=False,
        dataset_file=dataset_file,
    )

    dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_health_sciences_string.parquet"

    encoder.fine_tune_hyperparameter_testing(
        batch_sizes,
        epochs=epochs,
        base_learning_rate=base_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        batch_start_index=batch_start_index,
        scale=scale,
        train_matryoshka=False,
        adaptive_layers=False,
        dataset_file=dataset_file,
    )

    dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_life_sciences_string.parquet"

    encoder.fine_tune_hyperparameter_testing(
        batch_sizes,
        epochs=epochs,
        base_learning_rate=base_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        batch_start_index=batch_start_index,
        scale=scale,
        train_matryoshka=False,
        adaptive_layers=False,
        dataset_file=dataset_file,
    )

    dataset_file = r"E:\HugeDatasetBackup\DATA_CITATION_GRABBER\dataset_triplets_new\split_triplets\triplets_physical_sciences_string.parquet"

    encoder.fine_tune_hyperparameter_testing(
        batch_sizes,
        epochs=epochs,
        base_learning_rate=base_learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        batch_start_index=batch_start_index,
        scale=scale,
        train_matryoshka=False,
        adaptive_layers=False,
        dataset_file=dataset_file,
    )

    # encoder.train_all_datasets()


