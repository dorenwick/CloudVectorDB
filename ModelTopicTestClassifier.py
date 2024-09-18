import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelTopicTestClassifier:
    """
    We need to fix this up for the cloud computing.


    """


    def __init__(self):
        self.models_to_test = [
            {
                "model_path": "C:\\Users\\doren\\PycharmProjects\\Cite_Grabber\\Models\\field_classification_model\\fine_tuned_model_field",
                "dataset_path": "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\datasets\\topic_classification_dataset_merged_Social_Sciences.csv"
            },
            {
                "model_path": "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\models\\topic_classification_dataset_Social_Sciences",
                "dataset_path": "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\datasets\\topic_classification_dataset_merged_Social_Sciences.csv"
            },
            {
                "model_path": "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\models\\topic_classification_dataset_Earth_and_Planetary_Sciences",
                "dataset_path": "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\datasets\\topic_classification_dataset_merged_Earth_and_Planetary_Sciences.csv"
            },
            {
                "model_path": "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\models\\topic_classification_dataset_Economics,_Econometrics_and_Finance",
                "dataset_path": "C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\datasets\\topic_classification_dataset_merged_Economics,_Econometrics_and_Finance.csv"
            }
        ]
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    def load_test_dataset(self, dataset_path):
        dataset = load_dataset('csv', data_files=dataset_path)
        return dataset['train'].select(range(min(10000, len(dataset['train']))))  # Select up to 10,000 examples

    def load_label_mappings(self, model_path):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        id2label = config.get('id2label', {})
        label2id = config.get('label2id', {})
        return {int(k): v for k, v in id2label.items()}, {v: int(k) for k, v in id2label.items()}

    def tokenize_and_prepare(self, examples, label2id):
        tokenized = self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized['labels'] = [label2id.get(label, -1) for label in examples['label']]
        return tokenized

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def test_model(self, model_info):
        print(f"\nTesting model: {os.path.basename(model_info['model_path'])}")
        model = AutoModelForSequenceClassification.from_pretrained(model_info['model_path'])
        id2label, label2id = self.load_label_mappings(model_info['model_path'])
        test_dataset = self.load_test_dataset(model_info['dataset_path'])

        prepared_dataset = test_dataset.map(
            lambda examples: self.tokenize_and_prepare(examples, label2id),
            batched=True,
            remove_columns=test_dataset.column_names
        )

        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        results = trainer.evaluate(prepared_dataset)
        print(f"Results for {os.path.basename(model_info['model_path'])}:")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"Precision: {results['eval_precision']:.4f}")
        print(f"Recall: {results['eval_recall']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")

    def run_tests(self):
        for model_info in self.models_to_test:
            try:
                self.test_model(model_info)
            except Exception as e:
                print(f"Error testing model {model_info['model_path']}: {str(e)}")


if __name__ == "__main__":
    tester = ModelTopicTestClassifier()
    tester.run_tests()