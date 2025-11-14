"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : main.py
"""

from termcolor import colored

from load_dataset import LoadDataset
from spec_augment import SpegAugment
from load_model import LoadModel
from cleanup import Cleanup
from model_train import Train
from model_test import Test

DATASET_NAME = "benchmark_musicbench_logspec"
#DATASET_NAME = "benchmark_fsl10k_logspec"
#MODEL_NAME = "KEYR2"
#MODEL_NAME = "CNN14"
#MODEL_NAME = "RESNET50"
#MODEL_NAME = "CNNEEF"
#MODEL_NAME = "CNN"
#MODEL_NAME = "LSTM"
MODEL_NAME = "CNNGRU"
NUM_EPOCHS = 100

if __name__ == "__main__":
    print(colored("[Paso 1] Cargando dataset...", 'yellow'))
    DATASET_PATH = "datasets//" + DATASET_NAME
    data_load = LoadDataset()
    data_load.setup(DATASET_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test, unique_labels = data_load.start()

    print(colored("\n[Paso 2] Aumentando datos...", 'yellow'))
    data_augment = SpegAugment()
    data_augment.setup(X_train, y_train, X_val, y_val, X_test, y_test)
    train_loader, val_loader, test_loader = data_augment.start()

    print(colored("\n[Paso 3] Cargando modelo...", 'yellow'))
    model_load = LoadModel()
    model, gradcam_layer = model_load.setup(MODEL_NAME)

    print(colored("\n[Paso 4] Limpiando logs antiguos...", 'yellow'))
    cleanup = Cleanup()
    cleanup.setup(MODEL_NAME, DATASET_NAME)
    cleanup.start()

    print(colored("\n[Paso 5] Entrenando modelo...", 'yellow'))
    model_train = Train()
    model_train.setup(model, MODEL_NAME, DATASET_NAME, NUM_EPOCHS, y_train, unique_labels, train_loader, val_loader)
    model_dict = model_train.start()

    print(colored("\n[Paso 6] Validando modelo...", 'yellow'))
    model_test = Test()
    model_test.setup(model, gradcam_layer, MODEL_NAME, DATASET_NAME, model_dict, test_loader)
    model_test.start()
