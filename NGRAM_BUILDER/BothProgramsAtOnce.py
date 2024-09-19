import multiprocessing
import subprocess

def run_script(script):
    subprocess.run(['python', script])

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=run_script, args=('AbstractDataConstructionMultiGPUOnly.py',))
    p2 = multiprocessing.Process(target=run_script, args=('AbstractEmbeddingGenerator.py',))
    p3 = multiprocessing.Process(target=run_script, args=('AbstractKeywordExtractor.py',))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()



# python BothProgramsAtOnce.py

#
# (cite_grabber) root@C.12549921:/$ python BothProgramsAtOnce.py
# CUDA available: True
# CUDA device count: 2
# CUDA device name: NVIDIA GeForce RTX 4090
# Using device: cuda
# Starting embedding generation...
# Processing files:   0%|                                                                                                                                                                                                                                                      | 0/2587 [00:00<?, ?it/s]
# CUDA available: True
# CUDA available: True
# CUDA device count: 2
# CUDA device name: NVIDIA GeForce RTX 4090
# Using device: cuda:1
# CUDA device count: 2
# CUDA device name: NVIDIA GeForce RTX 4090
# num_gpus:  2
# Using 2 GPU(s): ['cuda:0', 'cuda:1']
# /root/miniconda3/envs/cite_grabber/lib/python3.10/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
#   warnings.warn(
# /root/miniconda3/envs/cite_grabber/lib/python3.10/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
#   warnings.warn(
# Starting keyword extraction...
# Processing files for keyword extraction:   0%|                                                                                                                                                                                                                               | 0/2587 [00:00<?, ?it/s]
# self.field_int_map:  {'id2label': {0: 'Economics, Econometrics and Finance', 1: 'Materials Science', 2: 'Environmental Science', 3: 'Medicine', 4: 'Psychology', 5: 'Dentistry', 6: 'Business, Management and Accounting', 7: 'Engineering', 8: 'Biochemistry, Genetics and Molecular Biology', 9: 'Agricultural and Biological Sciences', 10: 'Energy', 11: 'Earth and Planetary Sciences', 12: 'Health Professions', 13: 'Chemistry', 14: 'Chemical Engineering', 15: 'Social Sciences', 16: 'Pharmacology, Toxicology and Pharmaceutics', 17: 'Arts and Humanities', 18: 'Mathematics', 19: 'Immunology and Microbiology', 20: 'Veterinary', 21: 'Decision Sciences', 22: 'Nursing', 23: 'Physics and Astronomy', 24: 'Neuroscience', 25: 'Computer Science'}, 'label2id': {'Economics, Econometrics and Finance': 0, 'Materials Science': 1, 'Environmental Science': 2, 'Medicine': 3, 'Psychology': 4, 'Dentistry': 5, 'Business, Management and Accounting': 6, 'Engineering': 7, 'Biochemistry, Genetics and Molecular Biology': 8, 'Agricultural and Biological Sciences': 9, 'Energy': 10, 'Earth and Planetary Sciences': 11, 'Health Professions': 12, 'Chemistry': 13, 'Chemical Engineering': 14, 'Social Sciences': 15, 'Pharmacology, Toxicology and Pharmaceutics': 16, 'Arts and Humanities': 17, 'Mathematics': 18, 'Immunology and Microbiology': 19, 'Veterinary': 20, 'Decision Sciences': 21, 'Nursing': 22, 'Physics and Astronomy': 23, 'Neuroscience': 24, 'Computer Science': 25}}
# Checking for missing parquet files and verifying row counts...
# All batch files from 0 to 2586 are present.
#
# Starting file processing...
# Processing files:   0%|                                                                                                                                                                                                                                                      | 0/2587 [00:00<?, ?it/s]
# Batches:   4%|_______
# #
#
#