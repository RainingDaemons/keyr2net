"""
@Date         : 24-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/modules
@File         : metrics.py
"""

import numpy as np

class Metrics():
    @staticmethod
    def accuracy(correct, total):
        """
            Accuracy
            
            correct: cantidad de aciertos
            total: cantidad total de ejemplos
        """
        if total == 0:
            return 0.0
        
        return correct / total * 100
    
    @staticmethod
    def f1score(y_true, y_pred, num_classes=24):
        """
            F1-Score

            y_true: tensor o lista con labels reales
            y_pred: tensor o lista con labels predichos
        """
        f1_scores = []

        for cls in range(num_classes):
            tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
            fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
            fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / num_classes
        return macro_f1 * 100

    @staticmethod
    def focusratio(heatmap, bins, threshold=0.5):
        """
            Focus Ratio

            Input
            heatmap: tensor o lista con mapa de calor de la gradcam
            bins: tensor o lista con bins que corresponden a la escala a predecir
            threshold: umbral en [0,1] aplicado al perfil de energía por fila determinar si un bin está activo

            Returns:
            fr: Focus Ratio
            AB_count: Total de pixeles activados en bins relevantes
            NAB_count: Total de pixeles activados en bins no relevantes
            AB_pixels: Arreglo de pixeles AB
            NAB_pixels: Arreglo de pixeles NAB
        """
        if heatmap.ndim != 2:
            raise ValueError("heatmap debe ser un array 2D")

        H, W = heatmap.shape

        hm = heatmap.astype(np.float32)

        # Normalizar heatmap a [0,1]
        hm -= hm.min()
        hm /= (hm.max() + 1e-8)

        # Activaciones pixel a pixel
        activated_pixels = hm > threshold   # (H, W)

        # Bins relevantes dentro de rango
        valid_bins = np.array([b for b in bins if 0 <= b < H], dtype=int)

        # Mascara de relevance
        relevant_mask = np.zeros((H, W), dtype=bool)
        relevant_mask[valid_bins, :] = True

        # Pixel AB = activo Y relevante
        AB_pixels = activated_pixels & relevant_mask

        # Pixel NAB = activo Y NO relevante
        NAB_pixels = activated_pixels & (~relevant_mask)

        AB_count = int(AB_pixels.sum())
        NAB_count = int(NAB_pixels.sum())
        total_count = AB_count + NAB_count

        # Focus ratio
        fr = AB_count / total_count if total_count > 0 else 0.0

        return fr, AB_count, NAB_count, AB_pixels, NAB_pixels, activated_pixels, relevant_mask
