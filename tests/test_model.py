import tensorflow as tf
from detector_neumonia import model_fun

def test_model_loading_returns_keras_model():
    m = model_fun()
    assert isinstance(m, tf.keras.Model)
