# CloudVectorDB
A script for running on very powerful cloud computer, for building a very large dataset of triplets, then training encoders, then building the embeddings with the encoder, then building the vectordb with the encoder. 

Installation Instructions for Ubuntu Server
1. Update and Upgrade System
sudo apt update && sudo apt upgrade -y
2. Install CUDA
Follow the official NVIDIA instructions to install CUDA on your Ubuntu server. The exact steps may vary depending on your Ubuntu version and desired CUDA version. Generally, it involves:

Verify you have a CUDA-capable GPU
Download and install the NVIDIA CUDA Toolkit
Set up the required environment variables

3. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
Follow the prompts to install Miniconda. After installation, close and reopen your terminal.
4. Create and Activate Conda Environment
conda create -n cite_grabber python=3.10
conda activate cite_grabber
5. Install PyTorch 2.4.0
conda install pytorch==2.4.0 torchvision torchaudio cudatoolkit=11.8 -c pytorch
6. Install Transformers 4.39.0
pip install transformers==4.39.0
7. Install Sentence-Transformers 3.0.1
pip install sentence-transformers==3.0.1
8. Install Additional Required Packages
pip install pandas numpy tqdm pyarrow
9. Set Up Project Directory
mkdir -p /workspace/data/input
mkdir -p /workspace/data/output
mkdir -p /workspace/models
10. Download Required Models
Download the necessary models (keyphrase model and embedding model) and place them in the /workspace/models directory.
11. Modify the Script

Remove all references to the acronym classifier from the script.
Update the paths in the script to match your setup:
input_dir = "/workspace/data/input"
output_dir = "/workspace/data/output"
keyphrase_model_path = "/workspace/models/keyphrase_model"
embedding_model_path = "/workspace/models/embedding_model"


12. Run the Script
python your_script_name.py
Note: Make sure to replace your_script_name.py with the actual name of your Python script.
Remember to adjust these instructions if you have specific requirements or if your setup differs from a standard Ubuntu server environment.