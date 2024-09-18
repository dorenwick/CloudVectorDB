import multiprocessing
from OptimizedCloudAbstractKeyPhraseMultiGPU import OptimizedCloudAbstractKeyPhraseMultiGPU
from AbstractDataConstructionMultiGPUOnly import AbstractDataConstructionMultiGPUOnly

def run_keyphrase_extraction():
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    keyphrase_model_path = "/workspace/models/models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec/snapshots/bfc31646972e22ebf331c2e877c30439f01d35b3"

    processor = OptimizedCloudAbstractKeyPhraseMultiGPU(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path,
        models_per_gpu=4  # Adjust this based on your GPU memory
    )
    processor.run()

def run_data_construction():
    input_dir = "/workspace"
    output_dir = "/workspace/data/output"
    keyphrase_model_path = "/workspace/models/models--tomaarsen--span-marker-bert-base-uncased-keyphrase-inspec/snapshots/bfc31646972e22ebf331c2e877c30439f01d35b3"
    embedding_model_path = "/workspace/models/models--Snowflake--snowflake-arctic-embed-xs/snapshots/86a07656cc240af5c7fd07bac2f05baaafd60401"

    processor = AbstractDataConstructionMultiGPUOnly(
        input_dir=input_dir,
        output_dir=output_dir,
        keyphrase_model_path=keyphrase_model_path,
        embedding_model_path=embedding_model_path,
        extract_keywords=False,
        generate_embeddings=True
    )
    processor.run()

if __name__ == "__main__":
    # Create two separate processes for each module
    keyphrase_process = multiprocessing.Process(target=run_keyphrase_extraction)
    data_construction_process = multiprocessing.Process(target=run_data_construction)

    # Start both processes
    keyphrase_process.start()
    data_construction_process.start()

    # Wait for both processes to complete
    keyphrase_process.join()
    data_construction_process.join()

    print("Both processes completed successfully.")