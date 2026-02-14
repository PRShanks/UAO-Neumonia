"""Módulo de preprocesamiento de imágenes radiográficas.

Realiza operaciones de normalización y transformación de imágenes para
prepararlas como entrada al modelo de detección de neumonía.
"""

import cv2
import numpy as np


def preprocess(array):
    """Preprocesa una imagen para el modelo de Deep Learning.
    
    Operaciones:
    - Redimensiona a 512x512 píxeles
    - Convierte a escala de grises
    - Aplica CLAHE para mejorar contraste
    - Normaliza valores a [0, 1]
    - Agrega dimensiones de canal y batch
    
    Args:
        array: Imagen en formato numpy array (BGR o RGB)
    
    Returns:
        numpy array: Imagen preprocesada lista para predicción (1, 512, 512, 1)
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255.0
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
