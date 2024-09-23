# N-gram Processing Setup on vast.ai Ubuntu 22.0

This README provides instructions for setting up the environment and running the n-gram processing script on a Ubuntu 22.0 server rented from vast.ai.

## Initial Setup

1. Connect to your vast.ai instance using SSH.

2. Update the system:
   ```
   sudo apt update && sudo apt upgrade -y
   ```

3. Install Python 3 and pip (if not already installed):
   ```
   sudo apt install python3 python3-pip -y
   ```

## Dependencies Installation

1. Install required system libraries:
   ```
   sudo apt install libopenblas-dev -y
   ```

2. Install required Python packages:
   ```
   pip3 install numpy pandas tqdm pyarrow fastparquet multiprocess
   ```

## Project Setup

1. Create a project directory:
   ```
   mkdir ngram_processing
   cd ngram_processing
   ```

2. Upload your Python script (e.g., `ngram_processor.py`) to this directory using SCP or any other file transfer method.

3. Create an input directory for your data:
   ```
   mkdir /workspace
   ```

4. Upload your input Parquet files to the `/workspace` directory.

## Running the Script

1. Ensure you're in the project directory:
   ```
   cd ~/ngram_processing
   ```

2. Run the script:
   ```
   python3 ngram_processor.py
   ```

## Notes

- The script is set up to use `/workspace` as the input directory. Make sure your Parquet files are in this location.
- Output files will be saved in `/workspace/data/output/`.
- Adjust the `batch_size` in the script if you encounter memory issues.
- The script uses multiprocessing. Adjust the number of processes in the `__main__` section if needed.

## Troubleshooting

- If you encounter memory errors, try reducing the `batch_size` in the script.
- For permission issues, ensure you have the necessary rights to read from `/workspace` and write to `/workspace/data/output/`.

Remember to stop your vast.ai instance when you're done to avoid unnecessary charges!