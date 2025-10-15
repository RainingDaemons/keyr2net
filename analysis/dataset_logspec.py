"""
@Date         : 14-10-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : analysis
@File         : dataset_logspec.py
"""

import pandas as pd
import librosa
import numpy as np
import torch
from termcolor import colored
from utils import Utils

class DatasetLOGSPEC():
    def __init__(self):
        self.SAMPLE_RATE = 48000
        self.HOP_LENGTH = 512
        self.BINS_PER_OCTAVE = 24
        self.N_FFT = 8192
        self.FMIN = 65
        self.FMAX = 2100
        self.N_BINS = None
        self.AUDIO_DURATION = 0
        self.FRAMES_LIMIT = 937
        self.DATASET_NAME = ""

    def pad_or_truncate(self, spec, frames_limit):
        T, F = spec.shape
        if T < frames_limit:
            pad = np.zeros((frames_limit - T, F))
            spec = np.vstack([spec, pad])
        elif T > frames_limit:
            spec = spec[:frames_limit, :]
        return spec
    
    def log_triangular_filterbank(self, sr, n_fft, fmin, fmax, bands_per_octave):
        n_bands = int(np.floor(bands_per_octave * np.log2(fmax / fmin)))
        self.N_BINS = n_bands

        # Frecuencias de los bordes
        freqs = fmin * 2.0 ** (np.arange(0, n_bands + 2) / bands_per_octave)
        centers = freqs[1:-1]

        filters = np.zeros((n_bands, n_fft // 2 + 1))
        fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)

        for i in range(n_bands):
            f_left, f_center, f_right = freqs[i], centers[i], freqs[i + 1]

            eps = 1e-9
            left_slope = (fft_freqs - f_left) / max(f_center - f_left, eps)
            right_slope = (f_right - fft_freqs) / max(f_right - f_center, eps)

            filters[i] = np.maximum(0, np.minimum(left_slope, right_slope))

        return filters

    def process_audio(self, path):
        try:
            y, sr = librosa.load(path, sr=self.SAMPLE_RATE, mono=True)

            # Asegurar duración fija
            target_len = int(self.SAMPLE_RATE * self.AUDIO_DURATION)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]

            # Calcular STFT magnitud
            S = np.abs(librosa.stft(y, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)) ** 2

            # Banco de filtros logarítmicos
            fb = self.log_triangular_filterbank(
                sr=self.SAMPLE_RATE,
                n_fft=self.N_FFT,
                fmin=self.FMIN,
                fmax=self.FMAX,
                bands_per_octave=self.BINS_PER_OCTAVE
            )

            # Aplicar filtro y devolver
            log_spec = np.dot(fb, S)

            # Aplicar compresión logaritmica
            log_spec = np.log1p(log_spec).T

            # Normalizar min-max
            log_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min() + 1e-6)
            log_spec = np.nan_to_num(log_spec)  # Reemplaza NaN o inf por 0
            
            # Ajustar a frames_limit
            log_spec = self.pad_or_truncate(log_spec, self.FRAMES_LIMIT)

            # Verificar forma
            assert log_spec.shape[0] == self.FRAMES_LIMIT, f"Frames incorrectos: {log_spec.shape[0]}"

            return torch.tensor(log_spec, dtype=torch.float32)
        except Exception as e:
            print(f"Error procesando {path}: {e}")
            return None

    def setup(self, DF_NAME):
        self.DATASET_NAME = DF_NAME

    def start(self):
        # Cargar dataset
        DATASET_PATH = "annotations//" + self.DATASET_NAME + ".csv"
        try:
            data = pd.read_csv(DATASET_PATH)
            print(colored(f"Dataset {DATASET_PATH} cargado correctamente\n", 'green'))
        except:
            print(colored(f"[!] Dataset cargado sin éxito\n", 'red'))

        rows = [row._asdict() for row in data.itertuples(index=False)]

        # Establecer la longitud maxima para audios del dataset
        self.AUDIO_DURATION = round(data["duration"].min(), 2) 

        # Llamar procesador de datos
        procesador = Utils()
        procesador.setup(self.process_audio, rows, self.DATASET_NAME, "logspec")
        procesador.start()
