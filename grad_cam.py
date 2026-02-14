"""Módulo de explicabilidad con Grad-CAM (Gradient-weighted Class Activation Mapping).

Implementa la técnica Grad-CAM para generar mapas de calor que visualizan
las regiones de la imagen que el modelo considera para su predicción.
"""

import cv2
import numpy as np
import tensorflow as tf


def grad_cam(array, model, last_conv_layer_name: str = "conv10_thisone"):
    """Genera un mapa de calor Grad-CAM para explicar predicciones del modelo.
    
    Args:
        array: Imagen de entrada (numpy array)
        model: Modelo Keras/TensorFlow entrenado
        last_conv_layer_name (str): Nombre de la última capa convolucional
    
    Returns:
        numpy array: Imagen con mapa de calor superpuesto (BGR)
    """
    # prepare image batch
    from preprocess_img import preprocess

    img = preprocess(array)

    # build a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    # compute the gradient of the top predicted class for our input image
    preds = model.predict(img)
    pred_index = np.argmax(preds[0])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    conv_outputs = conv_outputs.numpy()[0]
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]
