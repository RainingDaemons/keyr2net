"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core
@File         : model_train.py
"""

import torch
import time
from collections import Counter
from termcolor import colored

from utils import ClassBalancedLoss, EarlyStopping
from metrics import Metrics
from logger import Logger

class Train():
    def __init__(self):
        self.model = None
        self.model_name = ""
        self.dataset_name = ""
        self.num_epochs = 0
        self.device = None

        self.y_train = None
        self.unique_labels = None
        self.train_loader = None
        self.val_loader = None

    def setup(self, MODEL, MODEL_NAME, DATASET_NAME, EPOCHS, Y_TRAIN, UNIQUE_LABELS, TRAIN_LOADER, VAL_LOADER):
        self.model = MODEL
        self.model_name = MODEL_NAME
        self.dataset_name = DATASET_NAME
        self.num_epochs = EPOCHS
        self.y_train = Y_TRAIN
        self.unique_labels = UNIQUE_LABELS
        self.train_loader = TRAIN_LOADER
        self.val_loader = VAL_LOADER

    def start(self):
        # Utilizar aceleración por GPU si está disponible
        print(colored(f"\nModelo cargado: {self.model_name}", 'blue'))

        if (torch.cuda.is_available()):
            print(colored("\n[!] Aceleración por GPU activada", 'blue'))
            self.device = torch.device("cuda")
        else:
            print(colored("\n[!] No se detecta GPU en este sistema", 'red'))
            self.device = torch.device("cpu")
        print(colored(f"Usando dispositivo: {self.device}\n", 'blue'))

        # ---------- Parámetros para Class-Balanced Loss ----------
        counts = Counter(self.y_train.tolist())
        num_classes = len(self.unique_labels)
        samples_per_cls = [counts[i] for i in range(num_classes)]

        # ---------- Función de perdida y optimizer ----------
        self.model.to(self.device)
        criterion = ClassBalancedLoss(samples_per_cls, num_classes=24, beta=0.9999).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        scaler = torch.amp.GradScaler()

        # Definir nombre a guardar del modelo
        new_name = self.model_name.lower() + "_" + self.dataset_name
        logger = Logger()
        logger.setup(new_name + "_train", "logs")

        start_time = time.time()
        for epoch in range(self.num_epochs):
            # ---------- Entrenamiento ----------
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            y_true_train, y_pred_train = [], []
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Utilizar precision mixta
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=self.device.type=='cuda'):
                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    predicted = torch.argmax(preds, dim=1) # clases predichas

                # Actualizar el scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                y_true_train.extend(batch_y.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())
                train_loss += loss.item()
                train_correct += (predicted == batch_y).sum().item() # aciertos
                train_total += batch_y.size(0) # total de ejemplos
            
            # Calcular metricas
            train_loss /= len(self.train_loader)
            train_acc = Metrics.accuracy(train_correct, train_total)
            train_f1 = Metrics.f1score(y_true_train, y_pred_train)

            # ---------- Validación ----------
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            y_true_val, y_pred_val = [], []
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    preds = self.model(batch_x)
                    loss = criterion(preds, batch_y)
                    predicted = torch.argmax(preds, dim=1) # clases predichas

                    y_true_val.extend(batch_y.cpu().numpy())
                    y_pred_val.extend(predicted.cpu().numpy())
                    val_loss += loss.item()
                    val_correct += (predicted == batch_y).sum().item() # aciertos
                    val_total += batch_y.size(0) # total de ejemplos

            # Calcular metricas
            val_loss /= len(self.val_loader)
            val_acc = Metrics.accuracy(val_correct, val_total)
            val_f1 = Metrics.f1score(y_true_val, y_pred_val)

            # Scheduler y EarlyStopping
            scheduler.step(val_loss)
            early_stopping(val_loss)

            # Logs
            logs = (
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%"
            )

            print(logs)
            logger.new_line(logs)

            if early_stopping.early_stop:
                print(colored(f"\n[!] Early stopping activado en epoch {epoch+1}", 'yellow'))
                break
        
        # Calcular tiempo
        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        logs = (f"\nElapsed time: {minutes}m {seconds}s")

        print(colored(logs, 'green'))
        logger.new_line(logs)

        # Guardar modelo
        new_name_model = new_name + ".pth"
        model_path = "trained//" + new_name_model
        try:
            torch.save(self.model.state_dict(), model_path)
            print(colored(f"\n[!] Modelo {new_name_model} guardado correctamente.", 'blue'))
            return new_name
        except:
            print(colored(f"\n[!] Error: No se pudo guardar el modelo {new_name_model}", 'red'))
            return None
