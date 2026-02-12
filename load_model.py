import tensorflow as tf

def load_trained_model(model_path: str):
    """Carga el modelo entrenado .h5."""
    model = tf.keras.models.load_model(model_path)
    return model

