#!/usr/bin/env python3
import sys
import os
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import matplotlib.cm as cm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load transformer model and extractor
extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.eval()

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    return img

def compute_attention_rollout(attentions):
    """
    Roll out attention from all transformer layers.
    attentions: list of (batch_size, num_heads, tokens, tokens)
    returns: (tokens,) attention from CLS to all patches
    """
    # average over heads
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for attn in attentions:
        attn_heads = attn.mean(dim=1)
        attn_heads = attn_heads + torch.eye(attn_heads.size(-1)).to(attn_heads.device)
        attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_heads, rollout)
    # Take [CLS] attention only
    mask = rollout[0, 0, 1:]  # skip CLS token
    return mask


def generate_attention_map_and_predict(image_path):
    try:
        original_img = load_and_preprocess_image(image_path)
        
        # Preprocess for ViT model
        inputs = extractor(images=original_img, return_tensors="pt")
        model.config.output_attentions = True  # Ensure attentions are returned
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            attentions = outputs.attentions  # List of attention tensors

        # Get predicted class
        predicted_index = logits.argmax(-1).item()
        predicted_class_name = model.config.id2label[predicted_index]

        # --- Attention Rollout ---
        def compute_attention_rollout(attentions):
            rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
            for attn in attentions:
                attn_heads = attn.mean(dim=1)
                attn_heads = attn_heads + torch.eye(attn_heads.size(-1)).to(attn_heads.device)
                attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
                rollout = torch.matmul(attn_heads, rollout)
            mask = rollout[0, 0, 1:]  # exclude CLS token
            return mask

        # Compute and normalize attention map
        attention_map = compute_attention_rollout(attentions)
        attention_map = attention_map.reshape(14, 14).detach().cpu().numpy()
        attention_map = attention_map / attention_map.max()

        # Resize to image size
        attention_map = np.uint8(255 * attention_map)
        attention_map = np.array(Image.fromarray(attention_map).resize(
            original_img.size, resample=Image.BICUBIC
        ))

        # Colorize and overlay
        heatmap_colored = cm.jet(attention_map)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)
        original_img_np = np.array(original_img)
        superimposed_img = heatmap_colored * 0.4 + original_img_np
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        # Save heatmap
        output_dir = os.path.dirname(image_path)
        visualization_path = os.path.join(output_dir, 'visualization.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(visualization_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return {
            "predicted_class_name": predicted_class_name,
            "visualization_path": visualization_path
        }

    except Exception as e:
        logging.error(f"Error generating attention map: {str(e)}")
        return {"error": str(e)}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python transformerScript.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image file not found at {image_path}"}))
        sys.exit(1)

    result = generate_attention_map_and_predict(image_path)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
