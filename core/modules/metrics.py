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
    def focusratio(heatmap, bins):
        """
            Focus Ratio

            heatmap: tensor o lista con mapa de calor de la gradcam
            bins: tensor o lista con bins que corresponden a la escala a predecir
        """
        axis: int = 0 # 0: filas (frecuencia), 1: columnas (tiempo)
        invert_vertical: bool = False # True si tus bins están definidos "desde abajo"
        normalize: bool = True
        aggregate_along_time: bool = True # sumar sobre tiempo (columnas) antes

        if heatmap.ndim != 2:
            raise ValueError("heatmap debe ser un array 2D")
        H, W = heatmap.shape

        hm = heatmap.astype(np.float32)
        if normalize:
            hm = hm - hm.min()
            hm = hm / (hm.max() + 1e-8)

        if axis == 0:
            valid = np.array([b for b in bins if 0 <= b < H], dtype=int)
            if invert_vertical:
                valid = (H - 1) - valid
            if aggregate_along_time:
                # Perfil de frecuencia (suma sobre tiempo)
                row_energy = hm.sum(axis=1)  # (H,)
                num = float(row_energy[valid].sum())
                den = float(row_energy.sum())
                return num / den if den > 0 else 0.0
            else:
                scale_mask = np.zeros_like(hm, dtype=bool)
                scale_mask[valid, :] = True
        else:
            valid = np.array([b for b in bins if 0 <= b < W], dtype=int)
            if aggregate_along_time:
                col_energy = hm.sum(axis=0)  # (W,)
                num = float(col_energy[valid].sum())
                den = float(col_energy.sum())
                return num / den if den > 0 else 0.0
            else:
                scale_mask = np.zeros_like(hm, dtype=bool)
                scale_mask[:, valid] = True

        num = float((hm * scale_mask).sum())
        den = float(hm.sum())
        return num / den if den > 0 else 0.0
