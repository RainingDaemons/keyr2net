"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : cleanup.py
"""

import os

class Cleanup():
    def __init__(self):
        self.log_base_name = ""

    def setup(self, MODEL_NAME, DATASET_NAME):
        self.log_base_name = MODEL_NAME + "_" + DATASET_NAME

    def start(self):
        logs_dir = "logs"

        # Archivos a eliminar
        files_to_delete = [
            f"{self.log_base_name}_train.txt",
            f"{self.log_base_name}_test.txt",
            f"{self.log_base_name}_bins.txt"
        ]

        for filename in files_to_delete:
            filepath = os.path.join(logs_dir, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Eliminado: {filepath}")
                except Exception as e:
                    print(f"No se pudo eliminar {filepath}: {e}")
            else:
                print(f"No existe: {filepath}")
