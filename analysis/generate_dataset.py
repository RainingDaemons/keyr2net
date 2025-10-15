"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : analysis
@File         : generate_dataset.py
"""

from termcolor import colored

from dataset_logspec import DatasetLOGSPEC

DATASET_NAME = "benchmark_fsl10k"
#DATASET_NAME = "benchmark_musicbench"
DATASET_TYPE = "LOGSPEC"

if __name__ == "__main__":
    if (DATASET_TYPE == "LOGSPEC"):
        print(colored(f"[!] Opción ingresada: {DATASET_TYPE}", 'yellow'))
        print(colored(f"Procesando {DATASET_NAME}\n", 'yellow'))
        data = DatasetLOGSPEC()
        data.setup(DATASET_NAME)
        data.start()
    else:
        print(colored(f"[!] Opción ingresada no válida\n", 'red'))
