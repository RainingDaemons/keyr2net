"""
@Date         : 24-11-2025
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
        self.file_name = ""
        self.list = ""
        self.log_path = ""
        self.ablation_test = ""

    def setup(self, FILE_NAME=None, ITEM_LIST=None, ABLATION_TEST=False):
        self.list = ITEM_LIST
        self.ablation_test = ABLATION_TEST

        if (self.ablation_test == False and FILE_NAME is not None):
            self.file_name = FILE_NAME.lower()
        if (self.ablation_test == False and FILE_NAME is None):
            self.file_name = "explainability"
        else:
            self.file_name = "ablation"
        
        self.log_path = os.path.join("results", self.file_name + "_final.txt")
        os.makedirs("results", exist_ok=True)

    def new_line(self, TEXT):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{TEXT}\n")

    def calc(self):
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        csv_path = os.path.join(base_path, "results", f"{self.file_name}.csv")

        df = pd.read_csv(csv_path)

        for i in range(len(self.list)):
            df_temp = ""
            if (self.ablation_test == False and self.file_name != "explainability"):
                df_temp = df[df["dataset"] == self.list[i]]
            if (self.ablation_test == False and self.file_name == "explainability"):
                df_temp = df[df["model"] == self.list[i].lower()]
            else:
                df_temp = df[df["model"] == self.list[i].lower()]

            # accuracy
            acc_mean = round(df_temp["test_accuracy"].mean(), 3)
            acc_std = round(df_temp["test_accuracy"].std(), 3)
            acc_var = round(df_temp["test_accuracy"].var(), 3)

            # macro f1 score
            f1_mean = round(df_temp["test_macrof1"].mean(), 3)
            f1_std = round(df_temp["test_macrof1"].std(), 3)
            f1_var = round(df_temp["test_macrof1"].var(), 3)

            # focus ratio
            fr_mean = round(df_temp["test_fr"].mean(), 3)
            fr_std = round(df_temp["test_fr"].std(), 3)
            fr_var = round(df_temp["test_fr"].var(), 3)
            
            # save info
            self.new_line(f"{self.list[i]}")
            self.new_line(f"Accuracy - Mean: {acc_mean} - Std: {acc_std} - Var: {acc_var}")
            self.new_line(f"Macro F1 Score - Mean: {f1_mean} - Std: {f1_std} - Var: {f1_var}")
            self.new_line(f"Focus Ratio - Mean: {fr_mean} - Std: {fr_std} - Var: {fr_var}\n")
