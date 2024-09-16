from sentence_transformers import SentenceTransformerTrainingArguments

# import tensorflow as tf
#
# for e in tf.compat.v1.train.summary_iterator("C:\\Users\\doren\\OneDrive\\Desktop\\DATA_CITATION_GRABBER\\models\\best_model_bs128_2024_08_12\\runs\\Aug12_13-59-37_Douglas\\events.out.tfevents.1723427979.Douglas.11416.0"):
#     for v in e.summary.value:
#         if v.tag == 'eval/loss':
#             print(f"Step {e.step}: {v.tag} = {v.simple_value}")

import pandas as pd



# Load the training arguments
training_args = SentenceTransformerTrainingArguments(r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_12\checkpoint-62\training_args.bin")

# training_args = SentenceTransformerTrainingArguments(r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_13\checkpoint-1556\training_args.bin")

# training_args = SentenceTransformerTrainingArguments(r"C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs2048_2024_08_13\checkpoint-2600\training_args.bin")

# "C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_12\runs\Aug12_13-59-37_Douglas\events.out.tfevents.1723427979.Douglas.11416.0"

# "C:\Users\doren\OneDrive\Desktop\DATA_CITATION_GRABBER\models\best_model_bs128_2024_08_12\runs\Aug12_13-59-37_Douglas\events.out.tfevents.1723427979.Douglas.11416.0"

# Now you can access the arguments
print(training_args)

# Access specific attributes
print(f"Learning rate: {training_args.learning_rate}")
print(f"Batch size: {training_args.per_device_train_batch_size}")