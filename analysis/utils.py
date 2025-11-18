"""
@Date         : 18-11-2025
@Author       : Felipe Gutiérrez Carilao
@Affiliation  : Universidad Andrés Bello
@Email        : f.gutierrezcarilao@uandresbello.edu
@Module       : analysis
@File         : utils.py
"""

import os
import torch
from termcolor import colored
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm

class Utils():
    def __init__(self):
        self.process_fn = None
        self.rows = None
        self.dataset_name = ""
        self.dataset_type = ""

    def procesar_fila(self, row_dict):
        try:
            dataset = row_dict['dataset']
            if dataset == "giantsteps":
                base_path = "./audio/giantsteps-mtg-key-dataset"
            elif dataset == "fsl10k":
                base_path = "./audio/fsl10k-dataset"
            elif dataset == "musicbench":
                base_path = "./audio/musicbench-dataset"
            else:
                return None

            audio_path = os.path.join(base_path, str(row_dict['file_directory']), str(row_dict['file_name']))
            if not os.path.exists(audio_path):
                print(f"Archivo no encontrado: {audio_path}")
                return None

            cqt = self.process_fn(audio_path)
            if cqt is None:
                return None

            # Transponer para (freq, time)
            return (cqt.T, row_dict['key'])

        except Exception as e:
            print(f"Error con {row_dict['file_name']}: {e}")
            return None

    def safe_procesar_fila(self, row_dict):
        try:
            return self.procesar_fila(row_dict)
        except Exception as e:
            print(f"Fallo con archivo {row_dict.get('file_name', 'unknown')}: {e}")
            return None
    
    def setup(self, PROCESS_FN, DATA, DATASET_NAME, DATASET_TYPE):
        self.process_fn = PROCESS_FN
        self.rows = DATA
        self.dataset_name = DATASET_NAME + "_" + DATASET_TYPE
        self.dataset_type = DATASET_TYPE

    def start(self):
        if (self.dataset_type == "logspec"):
            # Procesamiento en parelelo
            with tqdm_joblib(tqdm(desc="Procesando audios", total=len(self.rows))):
                results = Parallel(n_jobs=os.cpu_count(), prefer="threads")(
                    delayed(self.safe_procesar_fila)(row) for row in self.rows
                )

            # Filtrar espectrogramas válidos
            logspec_spectrograms = []
            labels = []

            for res in results:
                if res is not None:
                    tensor, label = res

                    # Chequeo de tensores válidos
                    if not torch.is_tensor(tensor):
                        print(f"Descartado (no es tensor): {label}")
                        continue
                    if tensor.numel() == 0:
                        print(f"Descartado (tensor vacío): {label}")
                        continue
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        print(f"Descartado (NaN/Inf en tensor): {label}")
                        continue

                    logspec_spectrograms.append(tensor)
                    labels.append(label)
            
            # Guardar dataset
            DATASET_PATH = "datasets//" + self.dataset_name
            chunk_size = 10000

            try:
                for i in range(0, len(logspec_spectrograms), chunk_size):
                    chunk_data = {
                        'logspec': torch.stack(logspec_spectrograms[i:i+chunk_size]),
                        'labels': labels[i:i+chunk_size]
                    }
                    chunk_file = f"{DATASET_PATH}_part{i//chunk_size}.pt"
                    torch.save(chunk_data, chunk_file, _use_new_zipfile_serialization=False)
                    print(colored(f"Chunk {chunk_file} guardado correctamente", 'blue'))
                
                print(colored(f"\nDataset guardado en {DATASET_PATH}_part*.pt", 'green'))
            except Exception as e:
                print(colored(f"\n[!] Error: No se pudo guardar el dataset {DATASET_PATH}", 'red'))
                print(e)
        else:
            print(colored(f"[!] Opción ingresada no válida", 'red'))
