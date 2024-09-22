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


