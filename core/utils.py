"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : utils.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, num_classes, beta=0.9999):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes  # normalización
        self.class_weights = torch.tensor(weights, dtype=torch.float)

    def forward(self, logits, targets):
        """
            logits: outputs raw del modelo
            targets: labels enteros
        """
        device = logits.device
        weights = self.class_weights.to(device)
        loss = F.cross_entropy(logits, targets, weight=weights)
        return loss

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # Activaciones y gradientes de la capa objetivo
        self.activations = None
        self.gradients = None

        # Hooks para capturar activaciones y gradientes
        self.fwd_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    
    def generate(self, input_tensor, class_idx=None, target_size=None):
        """
            Genera el mapa de calor Grad-CAM para una predicción.

            Args:
                input_tensor (torch.Tensor): entrada [B, C, H, W]
                class_idx (int): índice de la clase objetivo. Si es None, se usa la clase predicha.

            Returns:
                heatmap (np.ndarray): mapa de calor [H, W] normalizado entre 0-1
        """
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward de la clase objetivo
        target = output[:, class_idx]
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Verificar
        if self.gradients is None:
            print("[!] GradCAM: No se registraron gradientes de target_layer")
            return None

        # Ajustar salida según dimensiones
        if self.gradients.dim() == 4:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        elif self.gradients.dim() == 3:
            self.activations = self.activations.permute(0, 2, 1).unsqueeze(-1)
            self.gradients = self.gradients.permute(0, 2, 1).unsqueeze(-1)
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)

        elif self.gradients.dim() == 2:
            weights = self.gradients.mean(dim=0, keepdim=True)
            cam = (self.activations * weights).sum(dim=1, keepdim=True)
            cam = cam.unsqueeze(-1).unsqueeze(-1)

        else:
            raise ValueError(f"[!] GradCAM: Dimensión de gradientes no soportada - {self.gradients.shape}")   

        cam = F.relu(cam)

        # Interpolar al tamaño deseado
        if target_size is None:
            target_size = input_tensor.shape[2:]
        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)

        # Normalizar
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam
    
    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

class ScaleBins:
    def calc(self, SCALE_NAME):
        scales = {
            "A#:major": ["A#", "C", "D", "D#", "F", "G", "A"],
            "A#:minor": ["A#", "C", "C#", "D#", "F", "F#", "G#"],
            "A:major": ["A", "B", "C#", "D", "E", "F#", "G#"],
            "A:minor": ["A", "B", "C", "D", "E", "F", "G"],
            "B:major": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
            "B:minor": ["B", "C#", "D", "E", "F#", "G", "A"],
            "C:major": ["C", "D", "E", "F", "G", "A", "B"],
            "C:minor": ["C", "D", "D#", "F", "G", "G#", "A#"],
            "C#:major": ["C#", "D#", "F", "F#", "G#", "A#", "C"],
            "C#:minor": ["C#", "D#", "E", "F#", "G#", "A", "B"],
            "D:major": ["D", "E", "F#", "G", "A", "B", "C#"],
            "D:minor": ["D", "E", "F", "G", "A", "A#", "C"],
            "D#:major": ["D#", "F", "G", "G#", "A#", "C", "D"],
            "D#:minor": ["D#", "F", "F#", "G#", "A#", "B", "C#"],
            "E:major": ["E", "F#", "G#", "A", "B", "C#", "D#"],
            "E:minor": ["E", "F#", "G", "A", "B", "C", "D"],
            "F:major": ["F", "G", "A", "A#", "C", "D", "E"],
            "F:minor": ["F", "G", "G#", "A#", "C", "C#", "D#"],
            "F#:major": ["F#", "G#", "A#", "B", "C#", "D#", "F"],
            "F#:minor": ["F#", "G#", "A", "B", "C#", "D", "E"],
            "G:major": ["G", "A", "B", "C", "D", "E", "F#"],
            "G:minor": ["G", "A", "A#", "C", "D", "D#", "F"],
            "G#:major": ["G#", "A#", "C", "C#", "D#", "F", "G"],
            "G#:minor": ["G#", "A#", "B", "C#", "D#", "E", "F#"]
        }

        bin_factor_per_note = {
            "A": 0,
            "A#": 2,
            "B": 4,
            "C": 6,
            "C#": 8,
            "D": 10,
            "D#": 12,
            "E": 14,
            "F": 16,
            "F#": 18,
            "G": 20,
            "G#": 22,
        }

        # Devolver bins asociados a las notas que pertenecen a la escala, considerando 5 octavas
        mapped = scales[SCALE_NAME]
        bins = []
        for i in range(5):
            for note in mapped:
                bins.append(bin_factor_per_note[note] + i * 24)

        bins.sort()

        return bins
