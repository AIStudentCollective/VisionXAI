# -*- coding: utf-8 -*-
"""Modified TF-Attention_Rollout for VisionXAI

Adapted from the original Colab notebook to work as part of a Flask API
"""

from transformers import TFViTForImageClassification, ViTFeatureExtractor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
import os
import json
import base64
from io import BytesIO

class ViTAttentionRollout:
    def __init__(self, model, discard_ratio=0.9):

        self.model = model
        self.discard_ratio = discard_ratio

    def get_attention_maps(self, input_tensor):

        outputs = self.model(input_tensor, output_attentions=True)
        attention_maps = outputs.attentions

        attention_maps = [tf.reduce_mean(att, axis=1) for att in attention_maps]
        return attention_maps

    def compute_rollout(self, attention_maps):

        result = tf.eye(attention_maps[0].shape[-1])

        for attention_map in attention_maps:
            flat = tf.sort(tf.reshape(attention_map, [-1]))
            threshold = flat[int(flat.shape[0] * self.discard_ratio)]
            mask = attention_map > threshold
            attention_map = attention_map * tf.cast(mask, attention_map.dtype)

            # Add residual connection
            residual = tf.eye(attention_map.shape[-1])
            attention_map = attention_map + residual

            # Normalize
            attention_map = attention_map / tf.reduce_sum(attention_map, axis=-1, keepdims=True)

            # Multiply with the previous rollout
            result = tf.matmul(attention_map, result)

        return result

    def visualize(self, input_tensor, image_size=(224, 224)):

        attention_maps = self.get_attention_maps(input_tensor)
        rollout = self.compute_rollout(attention_maps)

        # Remove CLS
        rollout = rollout[0, 1:, 1:]

        num_patches = rollout.shape[0]
        grid_size = int(np.sqrt(num_patches))

        if grid_size ** 2 != num_patches:
            raise ValueError(
                f"Number of patches ({num_patches}) does not form a perfect square. Check input image size or "
                f"model patch size."
            )

        # turn into single map
        attention_map = tf.reduce_sum(rollout, axis=0)
        attention_map = tf.reshape(attention_map, (grid_size, grid_size))

        attention_map = tf.image.resize(
            tf.expand_dims(tf.expand_dims(attention_map, 0), -1),
            image_size,
            method='bilinear'
        )[0, :, :, 0]

        return attention_map.numpy()


def preprocess_image(image_path, feature_extractor):

    img = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=img, return_tensors="tf")
    return inputs["pixel_values"]


def visualize_vit_attention(model, feature_extractor, image_path, image_size=(224, 224)):
    input_tensor = preprocess_image(image_path, feature_extractor)
    attention_rollout = ViTAttentionRollout(model)
    attention_map = attention_rollout.visualize(input_tensor, image_size)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    heatmap = np.uint8(255 * attention_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    img = Image.open(image_path).convert("RGB")
    original_img = np.array(img.resize(image_size))
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Vision Transformer Attention Visualization')
    plt.show()


def save_attention_map(model, feature_extractor, image_path, output_path, image_size=(224, 224)):
    input_tensor = preprocess_image(image_path, feature_extractor)
    attention_rollout = ViTAttentionRollout(model)
    attention_map = attention_rollout.visualize(input_tensor, image_size)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    heatmap = np.uint8(255 * attention_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    img = Image.open(image_path).convert("RGB")
    original_img = np.array(img.resize(image_size))
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    # Save instead of displaying
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Get prediction
    inputs = feature_extractor(images=img, return_tensors="tf")
    outputs = model(inputs['pixel_values'])
    predicted_class_idx = tf.argmax(outputs.logits, axis=1).numpy()[0]
    
    # Get imagenet class labels
    class_name = model.config.id2label[predicted_class_idx]
    
    return class_name, output_path

# This function is used when the script is called directly by app.py
def main():
    if len(sys.argv) < 4:
        print("Error: Missing arguments", file=sys.stderr)
        sys.exit(1)
        
    # Parse arguments - note that we don't actually use class_names_path and weights_path 
    # for the pretrained transformer, but app.py passes them
    _, _, image_path = sys.argv[1:4]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'attention_map.png')
    
    # Load model and feature extractor
    try:
        model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Generate visualization
    try:
        class_name, output_path = save_attention_map(model, feature_extractor, image_path, output_path)
        # Print result in the format expected by app.py: "class_name,output_image_path"
        print(f"{class_name},{output_path}")
    except Exception as e:
        print(f"Error generating visualization: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()