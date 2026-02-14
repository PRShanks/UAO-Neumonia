"""Módulo para cargar modelos de Deep Learning entrenados.

Expone funciones para cargar y acceder al modelo de detección de neumonía
entrenado, basado en arquitectura Keras/TensorFlow.
"""

import tensorflow as tf


def load_trained_model(model_path: str):
    """Carga un modelo de Keras desde archivo .h5.
    
    Args:
        model_path (str): Ruta al archivo .h5 del modelo entrenado
    
    Returns:
        tf.keras.Model: Modelo cargado listo para predicciones
    """
    model = tf.keras.models.load_model(model_path)
    return model

