"""Módulo para lectura de imágenes radiográficas en formatos DICOM y JPG.

Proporciona funciones para cargar archivos médicos DICOM y imágenes JPG,
convirtiéndolas a formato compatible con el procesamiento del modelo.
"""

from PIL import Image
import numpy as np
import cv2
import pydicom as dicom


def read_dicom_file(path):
    """Lee un archivo DICOM y retorna arrays procesados.
    
    Args:
        path (str): Ruta al archivo DICOM
    
    Returns:
        tuple: (imagen RGB procesada, imagen PIL para visualización)
    """
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show



def read_jpg_file(path):
    """Lee un archivo JPG y retorna arrays procesados.
    
    Args:
        path (str): Ruta al archivo JPG
    
    Returns:
        tuple: (imagen procesada, imagen PIL para visualización)
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show
