import tensorflow as tf
import numpy as np
import matplotl ib.pyplot as plt
import cv2
from transformers import AutoFeatureExtractor, TFViTForImageClassification
from PIL import Image
import sys
import os

class TransformerAttentionPredictor:
    def __init__(self, model_name='google/vit-base-patch16-224', custom_weights=None):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        if custom_weights:
            self.model = TFViTForImageClassification.from_pretrained(custom_weights, output_attentions=True)
        else:
            self.model = TFViTForImageClassification.from_pretrained(model_name, output_attentions=True)

        self.labels = self.model.config.id2label

    def predict_and_visualize(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors='tf')
        outputs = self.model(inputs, training=False)

        logits = outputs.logits
        attentions = outputs.attentions

        predictions = tf.nn.softmax(logits, axis=-1)
        predicted_class_idx = tf.argmax(predictions[0]).numpy()
        predicted_class = self.labels[predicted_class_idx]

        heatmap = self._generate_attention_heatmap(attentions, image.size)
        heatmap_path = os.path.join(os.path.dirname(image_path), 'heatmap.png')
        cv2.imwrite(heatmap_path, heatmap)

        return predicted_class, heatmap_path

    def _generate_attention_heatmap(self, attentions, image_size):
        attention_arrays = [np.mean(np.array(attn), axis=1) for attn in attentions]
        num_tokens = attention_arrays[0].shape[-1]

        residual = np.eye(num_tokens)
        for attention in attention_arrays:
            attention += np.eye(num_tokens)
            attention /= attention.sum(axis=-1, keepdims=True)
            residual = np.matmul(attention[0], residual)

        relevance = residual[0, 1:]
        grid_size = int(np.sqrt(len(relevance)))
        heatmap = relevance.reshape(grid_size, grid_size)

        heatmap_resized = cv2.resize(heatmap, image_size, interpolation=cv2.INTER_LINEAR)
        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
        heatmap_color = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap_color


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python transformerScript.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predictor = TransformerAttentionPredictor(model_name='google/vit-base-patch16-224')
    try:
        predicted_class, heatmap_path = predictor.predict_and_visualize(image_path)
        print(f"{predicted_class}, {heatmap_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
