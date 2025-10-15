"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : model_train.py
"""

import torch
from sklearn.model_selection import train_test_split
from termcolor import colored

class LoadDataset:
    def __init__(self):
        self.DATASET_NAME = ""
        self.NUM_PARTS = 0

        self.X_train = ""
        self.y_train = ""
        self.X_val = ""
        self.y_val = ""
        self.X_test = ""
        self.y_test = ""
        self.unique_labels = ""

    def load(self):
        parts = []
        labels = []
        X = ""
        
        # Obtener etiquetas y características
        if (self.NUM_PARTS > 1):
            for i in range(self.NUM_PARTS):
                data = torch.load(f"{self.DATASET_NAME}_part{i}.pt")
                parts.append(data['logspec'])
                labels.extend(data['labels'])
            
            X = torch.cat(parts, dim=0)
        else:
            data = torch.load(f"{self.DATASET_NAME}_part0.pt")
            X = data['logspec']
            labels = data['labels']

        # Normalizar entre 0 y 1
        X = (X - X.min()) / (X.max() - X.min())

        # Crear índice de etiquetas
        unique_labels = sorted(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        y = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)

        # Asignar al conjunto de prueba el 20%
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        # Separar conjunto de entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=123, stratify=y_train_val
        )

        # Guardar dataset procesado
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.unique_labels = unique_labels

    def setup(self, NAME):
        self.DATASET_NAME = NAME

        if ("fsl10k" in NAME):
            self.NUM_PARTS = 1
        elif ("musicbench" in NAME):
            self.NUM_PARTS = 3

    def start(self):
        self.load()

        print(self.X_train.shape, self.y_train.shape)
        print(self.X_val.shape, self.y_val.shape)
        print(self.X_test.shape, self.y_test.shape)
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.unique_labels
