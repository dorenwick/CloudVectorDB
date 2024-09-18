import multiprocessing
import subprocess

# python BothProgramsAtOnce.py

def run_script(script):
    subprocess.run(['python', script])


if __name__ == '__main__':
    # TODO: We could make a third program here. We are still not utilizing the 2nd rtx 4090.

    p1 = multiprocessing.Process(target=run_script, args=('AbstractDataConstructionMultiGPUOnly.py',))
    p2 = multiprocessing.Process(target=run_script, args=('AbstractEmbeddingGenerator.py',))

    p1.start()
    p2.start()

    p1.join()
    p2.join()