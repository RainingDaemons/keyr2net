"""
@Date         : 18-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/modules
@File         : stats.py
"""

import os
import pandas as pd

class Stats():
    def __init__(self):
        self.model_name = ""

    def setup(self, MODEL_NAME, DATASETS):
        self.model_name = MODEL_NAME.lower()
        self.datasets = DATASETS
        self.log_path = os.path.join("results", self.model_name + "_final.txt")

        os.makedirs("results", exist_ok=True)

    def new_line(self, TEXT):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{TEXT}\n")

    def calc(self):
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        csv_path = os.path.join(base_path, "results", f"{self.model_name}.csv")

        df = pd.read_csv(csv_path)

        for i in range(len(self.datasets)):
            df_temp = df[df["dataset"] == self.datasets[i]]

            # accuracy
            acc_mean = round(df_temp["test_accuracy"].mean(), 3)
            acc_std = round(df_temp["test_accuracy"].std(), 3)
            acc_var = round(df_temp["test_accuracy"].var(), 3)

            # macro f1 score
            f1_mean = round(df_temp["test_macrof1"].mean(), 3)
            f1_std = round(df_temp["test_macrof1"].std(), 3)
            f1_var = round(df_temp["test_macrof1"].var(), 3)
            
            # save info
            self.new_line(f"{self.datasets[i]}")
            self.new_line(f"Accuracy - Mean: {acc_mean} - Std: {acc_std} - Var: {acc_var}")
            self.new_line(f"Macro F1 Score - Mean: {f1_mean} - Std: {f1_std} - Var: {f1_var}\n")
