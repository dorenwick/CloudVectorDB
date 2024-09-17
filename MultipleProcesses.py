

import subprocess



class MultipleProcesses():
    """


    """

    def __init__(self,):
        pass

    def run_script(self, script_name):
        subprocess.Popen(['python', script_name])



if __name__ == "__main__":
    multiprocess = MultipleProcesses()

    run_script('AbstractDataConstructionMultiGPU.py')
    run_script('script2.py')
