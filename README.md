# KeyR2Net: Una nueva arquitectura neuronal para la estimación de claves musicales

Este proyecto fue realizado dentro de mi tesis de pregado para optar al título de Ingeniero Civil Informático.

## Objetivos

Desarrollar un modelo de aprendizaje automático para la tarea de estimación de claves musicales que tenga una precisión comparable a modelos del estado del arte como son ResNet50, CNN14 y CNN-EFF.

## Organización del proyecto

Este proyecto está organizado a través de la siguiente estructura de carpetas:

| Carpeta | Contenido | Propósito |
| :--- | :--- | :--- |
| **`analysis/`** | Jupyter Notebooks (`.ipynb`) y código Python (`.py`) | Análisis exploratorio de los datos, transformación de audio en log-magnitude spectrograms (logspec) para el entrenamiento de los modelos |
| **`annotations/`** | Archivos delimitados por comas (`.csv`) | Anotaciones de datos relevantes para posteriomente crear los datasets |
| **`audio/`** | Audios (`.wav`) | Datos de los datasets originales de estudio (F10 y MB) sin procesar |
| **`core/`** | Código Python (`.py`) | Módulos principales del proyecto y pipeline de carga, entrenamiento y testeo de modelos |
| **`datasets/`** | Archivos en formato torch (`.pt`) | Datasets ya procesados listos para el entrenamiento de modelos |
| **`gradcam/`** | Imágenes (`.png`) | Comparación del input y mapa de calor obtenido por la GradCAM para una clase predicha |
| **`logs/`** | Archivos de texto plano (`.txt`) | Logs con resultados del entrenamiento y testeo de los modelos |
| **`trained/`** | Archivos del modelo entrenado (`.pth`) | Repositorio para guardar y cargar distinas versiones de los clasificadores |

## Tecnologías utilizadas

- Python
- Torch
- Torchaudio, Librosa
- Pandas, Numpy, Matplotlib
- Seaborn, Scikit-learn

## Licencia

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
