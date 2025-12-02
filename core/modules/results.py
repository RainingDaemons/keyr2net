"""
@Date         : 24-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/modules
@File         : results.py
"""

import os

class Results():
    def __init__(self):
        self.log_path = ""
        self.ablation_test = ""
    
    def setup(self, MODEL_NAME=None, ABLATION_TEST=False):
        FILE_NAME = ""
        header_line = ""
        self.ablation_test = ABLATION_TEST

        if (self.ablation_test == False and MODEL_NAME is not None):
            header_line = "dataset,seed,test_accuracy,test_macrof1,test_fr\n"
            FILE_NAME = MODEL_NAME.lower()
        if (self.ablation_test == False and MODEL_NAME is None):
            header_line = "model,seed,test_accuracy,test_macrof1,test_fr\n"
            FILE_NAME = "explainability"
        else:
            header_line = "model,seed,test_accuracy,test_macrof1,test_fr\n"
            FILE_NAME = "ablation"

        DEST_DIR = "results"
        self.log_path = os.path.join(DEST_DIR, FILE_NAME + ".csv")

        os.makedirs(DEST_DIR, exist_ok=True)

        # Verificar si la línea ya existe
        file_exists = os.path.isfile(self.log_path)
        header_present = False

        if file_exists:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() == header_line.strip():
                        header_present = True
                        break

        # Escribir la línea solo si no está presente
        if not header_present:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(header_line)
    
    def save(self, NAME, SEED, RESULT_ACC, RESULT_F1, RESULT_FR):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{NAME},{SEED},{RESULT_ACC},{RESULT_F1},{RESULT_FR}\n")
