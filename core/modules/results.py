"""
@Date         : 18-11-2025
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
    
    def setup(self, MODEL_NAME):
        DEST_DIR = "results"
        FILE_NAME = MODEL_NAME.lower()
        self.log_path = os.path.join(DEST_DIR, FILE_NAME + ".csv")

        os.makedirs(DEST_DIR, exist_ok=True)

        # Setup line
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"dataset,seed,test_accuracy,test_macrof1\n")
    
    def save(self, DATASET_NAME, SEED, RESULT_ACC, RESULT_F1):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{DATASET_NAME},{SEED},{RESULT_ACC},{RESULT_F1}\n")
