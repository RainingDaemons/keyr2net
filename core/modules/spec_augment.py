"""
@Date         : 18-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : core/modules
@File         : spec_augment.py
"""

import random
from torch.utils.data import DataLoader, Dataset

class KeyDataset(Dataset):
    def __init__(self, X, y, train=True, apply_specaugment=True, spec_augment_fn=None):
        self.X = X
        self.y = y
        self.train = train
        self.apply_specaugment = apply_specaugment
        self.spec_augment_fn = spec_augment_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Si es entrenamiento, aplicar SpecAugment
        if self.train and self.apply_specaugment and self.spec_augment_fn is not None:
            x = self.spec_augment_fn(x)

        x = x.unsqueeze(0)

        return x, y

class SpegAugment:
    def __init__(self):
        self.batch_size = 32
        self.num_workers = 0

        self.X_train = ""
        self.y_train = ""
        self.X_val = ""
        self.y_val = ""
        self.X_test = ""
        self.y_test = ""

    def setup(self, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, X_TEST, y_TEST):
        self.X_train = X_TRAIN
        self.y_train = Y_TRAIN
        self.X_val = X_VAL
        self.y_val = Y_VAL
        self.X_test = X_TEST
        self.y_test = y_TEST

    def oversampling(self, mel_spec, freq_mask_param=4, time_mask_param=8, num_masks=2):
        """
            Spec Augmentation

            mel_spec: Tensor [freq, time]
            freq_mask_param: tamaño máximo de bandas de frecuencia a borrar
            time_mask_param: tamaño máximo de bandas de tiempo a borrar
            num_masks: cantidad de máscaras a aplicar
        """
        spec = mel_spec.clone()

        # Frequency masking
        num_mel_channels = spec.shape[0]
        for _ in range(num_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, num_mel_channels - f)
            spec[f0:f0+f, :] = 0

        # Time masking
        num_frames = spec.shape[1]
        for _ in range(num_masks):
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, num_frames - t)
            spec[:, t0:t0+t] = 0

        return spec
    
    def start(self):
        train_dataset = KeyDataset(self.X_train, self.y_train, train=True, apply_specaugment=True, spec_augment_fn=self.oversampling)
        val_dataset = KeyDataset(self.X_val, self.y_val, train=False)
        test_dataset = KeyDataset(self.X_test, self.y_test, train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Verificar tamaños de los conjuntos de entrenamiento y prueba
        for x_batch, y_batch in train_loader:
            print(x_batch.shape, y_batch.shape)
            break

        print(len(train_dataset), len(val_dataset), len(test_dataset))

        return train_loader, val_loader, test_loader
