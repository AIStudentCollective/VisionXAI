import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import AutoFeatureExtractor, TFViTForImageClassification
from PIL import Image
import sys
import os
import uuid  # ✅ Added to generate unique heatmap filenames

# Ensure the heatmap directory exists
HEATMAP_FOLDER = "heatmaps"
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

class TransformerAttentionPredictor:
    def __init__(self, model_name='google/vit-base-patch16-224', custom_weights=None):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        if custom_weights:
            self.model = TFViTForImageClassification.from_pretrained(custom_weights, output_attentions=True)
        else:
            self.model = TFViTForImageClassification.from_pretrained(model_name, output_attentions=True)

        self.labels = self.model.config.id2label

    def predict_and_visualize(self, image_path):
        """Runs inference on an image and generates an attention rollout heatmap."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors='tf')
        outputs = self.model(inputs, training=False)

        logits = outputs.logits
        attentions = outputs.attentions  # Extract self-attention maps

        # Get prediction class
        predictions = tf.nn.softmax(logits, axis=-1)
        predicted_class_idx = tf.argmax(predictions[0]).numpy()
        predicted_class = self.labels[predicted_class_idx]

        # Generate the attention rollout heatmap
        heatmap = self._generate_attention_heatmap(attentions, image.size)

        # Save heatmap with a unique filename to prevent overwrites
        unique_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(HEATMAP_FOLDER, unique_filename)
        cv2.imwrite(heatmap_path, heatmap)

        return predicted_class, heatmap_path

    def _generate_attention_heatmap(self, attentions, image_size):
        """Computes the attention rollout heatmap from transformer attentions."""
        attention_arrays = [np.mean(np.array(attn), axis=1) for attn in attentions]
        num_tokens = attention_arrays[0].shape[-1]  # Number of patches (e.g., 197 for ViT-B/16)

        residual = np.eye(num_tokens)  # Identity matrix to initialize accumulation
        for attention in attention_arrays:
            attention += np.eye(num_tokens)  # Add skip connections
            attention /= attention.sum(axis=-1, keepdims=True)  # Normalize
            residual = np.matmul(attention[0], residual)  # Rollout through layers

        # Remove CLS token and reshape the attention map
        relevance = residual[0, 1:]
        grid_size = int(np.sqrt(len(relevance)))  # E.g., 14x14 for ViT-B/16
        heatmap = relevance.reshape(grid_size, grid_size)

        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, image_size, interpolation=cv2.INTER_LINEAR)

        # Normalize and apply colormap
        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
        heatmap_color = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return heatmap_color

# ✅ New function for Flask API integration
def process_image_attention(image_path):
    """
    Flask-compatible function to process an image with attention rollout.
    Returns prediction and heatmap URL.
    """
    predictor = TransformerAttentionPredictor(model_name='google/vit-base-patch16-224')
    try:
        predicted_class, heatmap_path = predictor.predict_and_visualize(image_path)
        return {"prediction": predicted_class, "heatmap_url": f"/heatmaps/{os.path.basename(heatmap_path)}"}
    except Exception as e:
        return {"error": str(e)}

# ✅ Command-line execution support for testing/debugging
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python attentionScript.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predictor = TransformerAttentionPredictor(model_name='google/vit-base-patch16-224')
    try:
        predicted_class, heatmap_path = predictor.predict_and_visualize(image_path)
        print(f"{predicted_class}, {heatmap_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
