"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : model_test.py
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from termcolor import colored

from metrics import Metrics
from logger import Logger
from utils import GradCAM, ScaleBins

class Test():
    def __init__(self):
        self.model = None
        self.gradcam_layer = None
        self.model_name = ""
        self.dataset_name = ""
        self.model_dict = ""
        self.test_loader = None
        self.device = None

    def setup(self, MODEL, GRADCAM_LAYER, MODEL_NAME, DATASET_NAME, MODEL_DICT, TEST_LOADER):
        self.model = MODEL
        self.gradcam_layer = GRADCAM_LAYER
        self.model_name = MODEL_NAME
        self.dataset_name = DATASET_NAME
        self.model_dict = "trained//" + MODEL_DICT + ".pth"
        self.test_loader = TEST_LOADER

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
        LOGGER_NAME = self.model_name.lower() + "_" + self.dataset_name + "_test"
        logger = Logger()
        logger.setup(LOGGER_NAME, "logs")

        # Inicializar GradCAM
        gradcam = GradCAM(self.model, target_layer=self.gradcam_layer)
        examples_per_class = {i: 0 for i in range(24)}
        max_examples = 10
        
        # Logear focus de bins
        scale = ScaleBins()
        scale_logger = Logger()
        SCALE_LOGGER_NAME = self.model_name.lower() + "_" + self.dataset_name + "_bins"
        scale_logger.setup(SCALE_LOGGER_NAME, "logs")

        os.makedirs(f"gradcam/{self.model_name.lower() + "_" + self.dataset_name}", exist_ok=True)

        # Evaluar modelo
        print(colored("\nEjecutando gradcam...", 'blue'))
        focus_total_ratio, focus_total_nonradio, focus_total = [], [], 0
        all_preds, all_outputs, all_labels = [], [], []
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
            for i in range(batch_x.size(0)):
                true_label = batch_y[i].item()
                pred_label = preds[i].item()

                # Guardar solo hasta 10 ejemplos por clase
                if examples_per_class[true_label] < max_examples:
                    SAVE_NAME = f"class_{true_label}_{examples_per_class[true_label]}"

                    # Guardar estado
                    was_training = self.model.training
                    self.model.eval()

                    # Deshabilitar cuDNN para RNN backward
                    cudnn_enabled = torch.backends.cudnn.enabled
                    torch.backends.cudnn.enabled = False

                    heatmap = gradcam.generate(batch_x[i].unsqueeze(0), class_idx=pred_label)

                    # Restaurar
                    torch.backends.cudnn.enabled = cudnn_enabled
                    self.model.train(was_training)

                    # ---------- Focus Bins ----------
                    if true_label == pred_label:
                        # Calcular bins relevantes
                        focus_bins = scale.calc(real_labels[pred_label])

                        # Aplicar umbral de intensidad (> 50%)
                        heatmap_np = heatmap.squeeze()
                        total_activation = heatmap_np.sum()

                        # Crear máscara de bins relevantes
                        focus_mask = np.zeros_like(heatmap_np, dtype=bool)
                        focus_mask[focus_bins, :] = True

                        # Activación en bins relevantes sobre el umbral
                        focus_activation = heatmap_np[focus_mask].sum()

                        # Activación en bins NO relevantes sobre el umbral
                        non_focus_mask = ~focus_mask
                        non_focus_activation = heatmap_np[non_focus_mask].sum()

                        # Calcular métricas
                        focus_ratio = focus_activation / total_activation
                        non_focus_ratio = non_focus_activation / total_activation

                        # Guardar totales
                        focus_total_ratio.append(focus_ratio)
                        focus_total_nonradio.append(non_focus_ratio)
                        focus_total += 1

                        logs = (
                            f"\nFile: {SAVE_NAME}.png\n"
                            f"Label: {real_labels[pred_label]}\n"
                            f"Focus Ratio: {focus_ratio:.2%}\n"
                            f"Non-Focus Ratio: {non_focus_ratio:.2%}\n"
                        )
                        scale_logger.new_line(logs)

                    # Guardar input y gradcam
                    input_img = batch_x[i].squeeze().cpu().numpy()
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(input_img, aspect="auto", origin="lower", cmap="magma")
                    plt.title(f"Input (True: {real_labels[true_label]}, Pred: {real_labels[pred_label]})")

                    plt.subplot(1, 2, 2)
                    plt.imshow(input_img, aspect="auto", origin="lower", cmap="magma")
                    plt.imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
                    plt.title("Grad-CAM")

                    save_path = os.path.join("gradcam", self.model_name.lower() + "_" + self.dataset_name, SAVE_NAME + ".png")
                    plt.savefig(save_path)
                    plt.close()

                    examples_per_class[true_label] += 1

        # Calcular promedio de focus bins
        avg_focus_ratio = Metrics.focus_ratio(focus_total_ratio, focus_total)
        avg_focus_nonratio = 1 - focus_ratio

        logs = (
            f"\nTotal bins: {focus_total}\n"
            f"Average Focus Ratio: {avg_focus_ratio:.2%}\n"
            f"Average Non-Focus Ratio: {avg_focus_nonratio:.2%}\n"
        )
        
        print(logs)
        scale_logger.new_line(logs)

        # Generar matriz de confusión y reporte de clasificación
        all_outputs = torch.cat(all_outputs).numpy()
        y_pred_classes = torch.cat(all_preds).numpy()
        y_true_classes = torch.cat(all_labels).numpy()

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
        plt.show()

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
