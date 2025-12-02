"""
@Date         : 24-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/modules
@File         : model_test.py
"""

import time
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from termcolor import colored

from .metrics import Metrics
from .logger import Logger
from .utils import GradCAM, ScaleBins

class Test():
    def __init__(self):
        self.model = None
        self.gradcam_layer = None
        self.model_name = ""
        self.dataset_name = ""
        self.model_dict = ""
        self.test_loader = None
        self.device = None
        self.ablation_test = None

    def setup(self, MODEL, GRADCAM_LAYER, MODEL_NAME, DATASET_NAME, DATASET_SEED, TEST_LOADER, ABLATION_TEST=False):
        self.model = MODEL
        self.gradcam_layer = GRADCAM_LAYER
        self.model_name = MODEL_NAME
        self.dataset_name = DATASET_NAME
        self.dataset_seed = str(DATASET_SEED)
        self.model_dict = f"trained/{self.model_name.lower()}/{self.dataset_name}/" + self.dataset_seed + ".pth"
        self.test_loader = TEST_LOADER
        self.ablation_test = ABLATION_TEST

    def start(self):
        real_labels = ['A#:major', 'A#:minor', 'A:major', 'A:minor',
        'B:major', 'B:minor', 'C#:major', 'C#:minor',
        'C:major', 'C:minor', 'D#:major', 'D#:minor',
        'D:major', 'D:minor', 'E:major', 'E:minor',
        'F#:major', 'F#:minor', 'F:major', 'F:minor',
        'G#:major', 'G#:minor', 'G:major', 'G:minor']

        if (torch.cuda.is_available()):
            print(colored("\n[!] Aceleración por GPU activada", 'blue'))
            self.device = torch.device("cuda")
        else:
            print(colored("\n[!] No se detecta GPU en este sistema", 'red'))
            self.device = torch.device("cpu")
        print(colored(f"Usando dispositivo: {self.device}", 'blue'))

        # Cargar modelo entrenado
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_dict, map_location=self.device))
        self.model.eval()

        # Definir nombre a guardar del modelo
        logger = Logger()
        logger.setup(self.dataset_seed + "_test", f"logs/{self.model_name.lower()}/{self.dataset_name}")

        # Inicializar GradCAM
        gradcam = GradCAM(self.model, target_layer=self.gradcam_layer)
        examples_per_class = {i: 0 for i in range(24)}
        max_examples = 10
        
        # Logear focus de bins
        scale = ScaleBins()
        scale_logger = Logger()
        scale_logger.setup(self.dataset_seed + "_bins", f"logs/{self.model_name.lower()}/{self.dataset_name}")

        os.makedirs(f"gradcam/{self.model_name.lower()}/{self.dataset_name}/{self.dataset_seed}", exist_ok=True)

        # Evaluar modelo
        print(colored("\nEjecutando gradcam...", 'blue'))
        fr_values, AB_values, NAB_values = [], [], []
        invalid_count, total_count = 0, 0
        all_preds, all_outputs, all_labels = [], [], []

        start_time = time.time()
        for batch_x, batch_y in self.test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward con mixed precision
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(batch_x)
                    preds = torch.argmax(outputs, 1)

            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())

            # ---------- Aplicar GradCAM ----------
            if (self.ablation_test == False):
                for i in range(batch_x.size(0)):
                    total_count += 1
                    true_label = batch_y[i].item()
                    pred_label = preds[i].item()

                    # Contabilizar hasta 10 ejemplos por clase
                    save_this = examples_per_class[true_label] < max_examples
                    if save_this:
                        SAVE_NAME = f"class_{true_label}_{examples_per_class[true_label]}"
                    
                    # Guardar estado
                    was_training = self.model.training
                    self.model.eval()

                    # Deshabilitar cuDNN para RNN backward
                    cudnn_enabled = torch.backends.cudnn.enabled
                    torch.backends.cudnn.enabled = False

                    # Grad-CAM
                    heatmap = gradcam.generate(batch_x[i].unsqueeze(0), class_idx=pred_label)

                    # Restaurar
                    torch.backends.cudnn.enabled = cudnn_enabled
                    self.model.train(was_training)

                    # ---------- Focus Bins ----------
                    scale_name_true = real_labels[true_label]
                    scale_bins_true = scale.calc(scale_name_true)

                    fr_result, AB, NAB, AB_pixels, NAB_pixels, activated_pixels, relevant_mask = Metrics.focusratio(heatmap, scale_bins_true)

                    # Excluir mapas gradcam sin información
                    if float(heatmap.var()) < 1e-12:
                        invalid_count += 1
                        fr_result = np.nan

                    fr_values.append(fr_result)
                    AB_values.append(AB)
                    NAB_values.append(NAB)

                    # Log por muestra
                    scale_logger.new_line(f"fr_result: {0.0 if np.isnan(fr_result) else fr_result:.4f}")

                    # Guardar imagenes de gradcam contabilizadas
                    if save_this:
                        # Obtener input
                        input_img = batch_x[i].squeeze().cpu().numpy()
                        H, W = heatmap.shape

                        overlay = np.zeros((H, W, 3), dtype=np.float32)

                        # Pixeles relevantes en azul (Scale Bins)
                        overlay[relevant_mask] = [0, 0, 1]

                        # Pixeles AB en verde
                        overlay[AB_pixels] = [0, 1, 0]

                        # Pixeles NAB en rojo
                        overlay[NAB_pixels] = [1, 0, 0]

                        # ----- Plot -----
                        plt.figure(figsize=(12, 6))

                        # Input image
                        plt.subplot(1, 2, 1)
                        plt.imshow(input_img, aspect="auto", origin="lower", cmap="magma")
                        plt.title(f"Input\nTrue: {real_labels[true_label]} | Pred: {real_labels[pred_label]}")

                        # Gradcam image colored
                        plt.subplot(1, 2, 2)
                        plt.imshow(input_img, aspect="auto", origin="lower", cmap="magma")
                        plt.imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.2)
                        plt.imshow(overlay, aspect="auto", origin="lower", alpha=0.8)
                        plt.title("Grad-CAM + AB (verde) / NAB (rojo) / Relevantes (azul)")

                        # Caption
                        plt.figtext(
                            0.5, 0.01,
                            f"AB = {AB:.4f} | NAB = {NAB:.4f} | FR = {fr_result:.4f}",
                            ha="center", fontsize=12
                        )

                        save_path = os.path.join("gradcam", self.model_name.lower(), self.dataset_name, self.dataset_seed, SAVE_NAME + ".png")
                        plt.savefig(save_path)
                        plt.close()

                        examples_per_class[true_label] += 1

        # Calcular estadísticas finales
        fr_mean = 0
        logs = ""
        if (self.ablation_test == False):
            def stats(arr):
                a = np.array(arr, dtype=float)
                a = a[~np.isnan(a)]
                if a.size == 0:
                    return 0.0, 0.0, 0.0, 0
                mean = a.mean()
                std = a.std(ddof=1) if a.size > 1 else 0.0
                var = a.var(ddof=1) if a.size > 1 else 0.0
                return mean, std, var, a.size

            fr_mean, fr_std, fr_var, n_valid = stats(fr_values)

            logs = (f"FR Stats (n={n_valid}/{total_count} -> Mean={fr_mean:.4f} - Std={fr_std:.4f} - Var={fr_var:.4f}")
        else:
            logs = (f"[!] Perfoming ablation test, skipping gradcam...")
        
        print(logs)
        scale_logger.new_line(logs)

        # Generar matriz de confusión y reporte de clasificación
        all_outputs = torch.cat(all_outputs).numpy()
        y_pred_classes = torch.cat(all_preds).numpy()
        y_true_classes = torch.cat(all_labels).numpy()

        cm_name = f"{self.model_name.lower()}/{self.dataset_name}/{self.dataset_seed}"
        os.makedirs(f"cm/{cm_name}", exist_ok=True)

        print(colored("Calculando resultados...", 'blue'))
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=real_labels))

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=real_labels, yticklabels=real_labels,
                    cmap="Blues")
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"cm/{cm_name}/confusion_matrix.png", dpi=300)

        # Mostrar métricas globales
        logger.new_line("\nComparación de etiquetas reales vs predichas:")
        logger.new_line("Índice | Real | Predicción | ¿Correcto?")
        logger.new_line("--------------------------------------")

        correct_count = 0
        for i in range(len(y_true_classes)):
            real = real_labels[y_true_classes[i]]
            pred = real_labels[y_pred_classes[i]]
            is_correct = real == pred

            if is_correct:
                correct_count += 1

            logs = (f"{i:6} | {real:8} | {pred:8} | {'✓' if is_correct else '✗'}")
            logger.new_line(logs)

        test_acc = Metrics.accuracy(correct_count, len(y_true_classes))
        test_f1 = Metrics.f1score(y_true_classes, y_pred_classes)

        print("\nResumen:")
        print(f"Total muestras: {len(y_true_classes)}")
        print(f"Correctas: {correct_count}")

        logs = (f"\nTest Acc: {test_acc:.2f}%, Test F1: {test_f1:.2f}%")

        print(logs)
        logger.new_line(logs)

        # Calcular tiempo
        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        logs = (f"\nElapsed time: {minutes}m {seconds}s")

        print(colored(logs, 'green'))
        logger.new_line(logs)

        return test_acc, test_f1, fr_mean
