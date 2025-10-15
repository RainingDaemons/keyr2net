"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : metrics.py
"""

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
    def focus_ratio(bins, total):
        """
            Focus Ratio
            
            bins: bins relevantes (pertencen a umbral de intesidad)
            total: cantidad total de bins relevantes
        """
        if total == 0:
            return 0.0
        
        return sum(bins) / total
