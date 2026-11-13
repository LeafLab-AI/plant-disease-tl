from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


class GradCAMExplainer:
    def __init__(self, model: keras.Model):
        self.model = model
        self.grad_model, self.layer_name = self._build_gradcam_model()

    def _build_gradcam_model(self) -> tuple[keras.Model, str]:
        """Extracts the final Conv2D layer dynamically to build the gradient-capturing model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return keras.Model(self.model.input, [layer.output, self.model.output]), layer.name
                
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                convs = [l for l in layer.layers if isinstance(l, keras.layers.Conv2D)]
                if convs:
                    conv = convs[-1]
                    return keras.Model(self.model.input, [conv.output, self.model.output]), conv.name
                    
        raise RuntimeError("No Conv2D layer found for Grad-CAM.")

    def generate_heatmap(self, img: np.ndarray, class_idx: int) -> np.ndarray:
        """Computes the Grad-CAM heatmap for a specific image and class."""
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(img[None, ...], training=False)
            loss = preds[:, class_idx]
            
        grads = tape.gradient(loss, conv_out)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        
        cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0].numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return tf.image.resize(cam[..., None], img.shape[:2]).numpy()[..., 0]