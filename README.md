# CloudVectorDB

A script for running on very powerful cloud computers,
building a large dataset of triplets,
training encoders, generating embeddings,
and constructing a vector database.

This is for a vast.ai server with multiple gpu's.  

## Installation Instructions for Ubuntu Server

1. **Update and Upgrade System**
   ```
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install CUDA**
   ```
   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
   sudo sh cuda_12.4.0_550.54.14_linux.run
   ```
   Follow the prompts to install. Make sure to say yes to adding CUDA to your PATH.

3. **Add CUDA to PATH and LD_LIBRARY_PATH**
   Add these lines to your `~/.bashrc` file:
   ```
   export PATH=/usr/local/cuda-12.4/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
   ```
   Then run:
   ```
   source ~/.bashrc
   ```

4. **Install Miniconda**
   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   Follow the prompts to install Miniconda. After installation, close and reopen your terminal.

5. **Create and Activate Conda Environment**
   ```
   conda create -n cite_grabber python=3.10
   conda activate cite_grabber
   ```

6. **Install PyTorch with CUDA support**
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

7. **Install Transformers**
   ```
   pip install transformers==4.39.0
   ```

8. **Install Sentence-Transformers**
   ```
   pip install sentence-transformers==3.0.1
   ```

9. **Install Additional Required Packages**
   ```
   pip install pandas numpy tqdm pyarrow span-marker
   ```

10. **Set Up Project Directory**
    ```
    mkdir -p /workspace/data/input
    mkdir -p /workspace/data/output
    mkdir -p /workspace/models
    ```

11. **Download Required Models**
    Download the necessary models (keyphrase model and embedding model) and place them in the `/workspace/models` directory.

12. **Verify CUDA Installation**
    Run these commands to verify CUDA is properly installed:
    ```
    nvcc --version
    python -c "import torch; print(torch.cuda.is_available())"
    ```
    They should show the CUDA version and `True` respectively.

13. **Modify the Script**
    - Update the paths in the script to match your setup:
      ```python
      input_dir = "/workspace/data/input"
      output_dir = "/workspace/data/output"
      keyphrase_model_path = "/workspace/models/keyphrase_model"
      embedding_model_path = "/workspace/models/embedding_model"
      ```

14. **Run the Script**
    ```
    python AbstractDataConstructionMultiGPU.py
    ```

Note: Make sure to adjust these instructions if you have specific requirements or if your setup differs from a standard Ubuntu server environment.