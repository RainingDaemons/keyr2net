"""
@Date         : 24-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : ablation_test.py
"""

from termcolor import colored

from modules.load_dataset import LoadDataset
from modules.spec_augment import SpegAugment
from modules.load_model import LoadModel
from modules.cleanup import Cleanup
from modules.model_train import Train
from modules.model_test import Test
from modules.results import Results
from modules.stats import Stats

DATASET_NAME = "musicbench_logspec"
MODEL_LIST = ["CNN",
            "LSTM",
            "CNNGRU"]
SEED_LIST = [123, 456, 789]
NUM_EPOCHS = 100

MODEL_TOTAL = len(MODEL_LIST)
SEED_TOTAL = len(SEED_LIST)

if __name__ == "__main__":
    print(colored(f"[!] Starting ablation test...\n", 'red'))

    results = Results()
    results.setup(None, True)

    stats = Stats()
    stats.setup(None, MODEL_LIST, True)

    for md_i in range(MODEL_TOTAL):
        MODEL_NAME = MODEL_LIST[md_i]
    
        for sd_i in range(SEED_TOTAL):
            SEED = SEED_LIST[sd_i]

            print(colored(f"=====================", 'red'))
            print(colored(f"[!] Model: {MODEL_NAME}", 'red'))
            print(colored(f"[!] Run: {sd_i+1}/{SEED_TOTAL}", 'red'))
            print(colored(f"[!] Seed: {SEED}", 'red'))
            print(colored(f"=====================\n", 'red'))

            print(colored("[Paso 1] Cargando dataset...", 'yellow'))
            DATASET_PATH = "datasets//" + DATASET_NAME
            data_load = LoadDataset()
            data_load.setup(DATASET_PATH, SEED)
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
            cleanup.setup(MODEL_NAME, DATASET_NAME, SEED)
            cleanup.start()

            print(colored("\n[Paso 5] Entrenando modelo...", 'yellow'))
            model_train = Train()
            model_train.setup(model, MODEL_NAME, DATASET_NAME, SEED, NUM_EPOCHS, y_train, unique_labels, train_loader, val_loader)
            model_train.start()

            print(colored("\n[Paso 6] Validando modelo...", 'yellow'))
            model_test = Test()
            model_test.setup(model, gradcam_layer, MODEL_NAME, DATASET_NAME, SEED, test_loader, True)
            test_acc, test_f1, test_fr = model_test.start()

            print(colored("\n[Paso 7] Guardando resultados...", 'yellow'))
            results.save(MODEL_NAME.lower(), SEED, test_acc, test_f1, test_fr)

    print(colored("\n[!] Calculando resultados finales...", 'red'))
    stats.calc()
    print(colored("\n[!] Benchmark finalizado...", 'green'))
