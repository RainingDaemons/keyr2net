"""
@Date         : 24-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/modules
@File         : cleanup.py
"""

import os
import shutil

class Cleanup():
    def __init__(self):
        self.model_name = ""
        self.dataset_name = ""
        self.dataset_seed = ""

    def setup(self, MODEL_NAME, DATASET_NAME, SEED):
        self.model_name = MODEL_NAME.lower()
        self.dataset_name = DATASET_NAME
        self.dataset_seed = str(SEED)

    def start(self):
        # Archivos a eliminar
        """
        files_to_delete = [
            os.path.join("logs", self.model_name, self.dataset_name, f"{self.dataset_seed}_train.txt"),
            os.path.join("logs", self.model_name, self.dataset_name, f"{self.dataset_seed}_test.txt"),
            os.path.join("logs", self.model_name, self.dataset_name, f"{self.dataset_seed}_bins.txt")
        ]
        """
        files_to_delete = [
            os.path.join("logs", self.model_name, self.dataset_name, f"{self.dataset_seed}_test.txt"),
            os.path.join("logs", self.model_name, self.dataset_name, f"{self.dataset_seed}_bins.txt")
        ]

        # Carpetas a eliminar
        dirs_to_delete = [
            os.path.join("cm", self.model_name, self.dataset_name, self.dataset_seed),
            os.path.join("gradcam", self.model_name, self.dataset_name, self.dataset_seed)
        ]

        for filepath in files_to_delete:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Eliminado: {filepath}")
                except Exception as e:
                    print(f"No se pudo eliminar {filepath}: {e}")
            else:
                print(f"No existe: {filepath}")
        
        for dirpath in dirs_to_delete:
            if os.path.exists(dirpath):
                try:
                    shutil.rmtree(dirpath)
                    print(f"Carpeta eliminada: {dirpath}")
                except Exception as e:
                    print(f"No se pudo eliminar {dirpath}: {e}")
            else:
                print(f"No existe: {dirpath}")
