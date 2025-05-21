from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io
import base64
import tempfile
import os
from importlib.util import spec_from_file_location, module_from_spec
import vit_attention_rollout

pytorch_bp = Blueprint('pytorch', __name__)

# Default PyTorch model layers for Grad-CAM
MODEL_TARGET_LAYERS = {
   "resnet18": "layer4",
    "resnet34": "layer4",
    "resnet50": "layer4",
    "resnet101": "layer4",
    "resnet152": "layer4",
    "alexnet": "features.11",
    "vgg11": "features.20",
    "vgg13": "features.20",
    "vgg16": "features.29",
    "vgg19": "features.29",
    "inception_v3": "Mixed_7c",
    "densenet121": "features.norm5",
    "densenet169": "features.norm5",
    "densenet201": "features.norm5",
    "densenet161": "features.norm5",
    "mobilenet_v2": "features.18",
    "shufflenet_v2_x0_5": "conv5",
    "shufflenet_v2_x1_0": "conv5"
}

class CustomModelWrapper(nn.Module):
    def __init__(self, model, target_layer_name):
        super(CustomModelWrapper, self).__init__()
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        target_layer = dict(self.model.named_modules()).get(target_layer_name)
        if not target_layer:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        self.model(x)
        return self.activations

def load_class_labels(csv_file):
    """Load class labels from CSV file."""
    df = pd.read_csv(csv_file)
    return dict(zip(df['index'], df['class_name']))

def load_custom_model(model_file, weights_file):
    """Dynamically loads a custom CNN model from uploaded files."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_model_file:
        temp_model_file.write(model_file.read())
        temp_model_file_path = temp_model_file.name

    try:
        spec = spec_from_file_location("user_model", temp_model_file_path)
        user_model = module_from_spec(spec)
        spec.loader.exec_module(user_model)

        model_classes = [name for name in dir(user_model) if not name.startswith("__")]
        if not model_classes:
            raise ValueError("No model class found.")

        model_class = getattr(user_model, model_classes[0])
        model = model_class()

        state_dict = torch.load(weights_file, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

    finally:
        os.remove(temp_model_file_path)

    return model

def find_last_conv_layer(model):
    """Finds the last convolutional layer in a CNN model."""
    conv_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No convolutional layer found.")
    return conv_layers[-1]

def adjust_fc_layer(model, input_image_size=(224, 224)):
    """Dynamically adjusts the FC layer if necessary."""
    dummy_input = torch.randn(1, 3, *input_image_size)
    with torch.no_grad():
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                break
            dummy_input = module(dummy_input)

    flattened_size = dummy_input.numel()
    fc_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not fc_layers:
        raise ValueError("No FC layer found.")

    last_fc = fc_layers[-1]
    output_features = getattr(model, last_fc).out_features
    setattr(model, last_fc, nn.Linear(flattened_size, output_features))
    
    return model

@pytorch_bp.route('/heatmap', methods=['POST'])
def heatmap():
    """Handles image processing and Grad-CAM visualization for models."""
    model_name = request.form.get('model_name')
    image_file = request.files.get('image_path')
    class_labels_file = request.files.get('class_labels_csv')
    custom_model_file = request.files.get('custom_model_file')  # Optional
    custom_weights_file = request.files.get('custom_weights_file')  # Optional

    if not image_file or not class_labels_file:
        return jsonify({"error": "Image and class labels CSV are required"}), 400

    class_labels = load_class_labels(class_labels_file)

    # **Check if using a custom model**
    if custom_model_file and custom_weights_file:
        print("Using custom model...")

        model = load_custom_model(custom_model_file, io.BytesIO(custom_weights_file.read()))
        target_layer = find_last_conv_layer(model)
        model = adjust_fc_layer(model)
        print(f"Using target layer '{target_layer}' for custom model")

    elif model_name in MODEL_TARGET_LAYERS:
        print(f"Using pre-trained model: {model_name}")
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        target_layer = MODEL_TARGET_LAYERS[model_name]
    else:
        return jsonify({"error": "Invalid model selection"}), 400

    wrapped_model = CustomModelWrapper(model, target_layer)
    wrapped_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    output = wrapped_model(input_tensor)
    probabilities = F.softmax(output, dim=1).detach().cpu().numpy()

    predicted_class_idx = output.argmax(dim=1).item()
    predicted_class_name = class_labels.get(predicted_class_idx, f"Unknown ({predicted_class_idx})")
    predicted_class_prob = probabilities[0, predicted_class_idx] * 100

    print(f"Predicted Class: {predicted_class_name} ({predicted_class_prob:.2f}%)")

    wrapped_model.zero_grad()
    output[0, predicted_class_idx].backward()

    gradients = wrapped_model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = wrapped_model.get_activations(input_tensor).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
    heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    original_img = np.array(img)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colormap, 0.5, 0)

    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": image_base64,
        "predicted_class": predicted_class_name,
        "predicted_probability": f"{predicted_class_prob:.2f}%",
        "target_layer_used": target_layer
    }), 200



# Acceptabble models include: 
# resnet18, resnet34, resnet50, resnet101, resnet152 --> with target layer4
# alexnet --> with target features.11 
# vgg11, vgg13 --> with target features.20 
# vgg16, vgg19 --> with target features.29 
# inception_v3 --> with target Mixed_7c 
# densenet121, densenet169, densenet201, densenet161 --> with target features.norm5
# mobilenet_v2 --> with target features.18  
# shufflenet_v2_x0_5, shufflenet_v2_x1_0 --> with target conv5
# mnasnet0_5, mnasnet1_0 --> with target layers[-1] --> NOT 

@pytorch_bp.route('/heatmap_vit', methods=['POST'])
def vit_heatmap():

    print('Received request for ViT heatmap generation.')

    model_name = request.form.get('model_name')
    weights_file = request.files.get('weights_path') 
    image_file = request.files['image_path']
    class_labels_file = request.files['class_labels_csv'] 
    num_classes = request.form.get('num_classes')
    image_size = request.form.get('image_size')

    print('Displaying request data:')
    print(f"Model Name: {model_name}")
    print(f"Weights File: {weights_file}")
    print(f"Image File: {image_file}")
    print(f"Class Labels File: {class_labels_file}")
    print(f"Num Classes: {num_classes}")
    print(f"Image Size: {image_size}")

    if not model_name or not image_file or not class_labels_file or not num_classes or not image_size:
        return jsonify("Unable to calculate attention rollout: insufficient data."), 500

    try:
        print('Attempting makeRollout()')
        response = vit_attention_rollout.makeRollout(
            model_name, 
            weights_file, 
            image_file, 
            class_labels_file, 
            num_classes, 
            image_size
        )
        if isinstance(response, Exception):
            raise response
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify('Error in attention rollout.'), 500

# Acceptable models include:
"""
vit_base_mci_224
vit_base_patch8_224
vit_base_patch14_dinov2
vit_base_patch14_reg4_dinov2
vit_base_patch16_18x2_224
vit_base_patch16_224
vit_base_patch16_224_miil
vit_base_patch16_384
vit_base_patch16_clip_224
vit_base_patch16_clip_384
vit_base_patch16_clip_quickgelu_224
vit_base_patch16_gap_224
vit_base_patch16_plus_240
vit_base_patch16_plus_clip_240
vit_base_patch16_reg4_gap_256
vit_base_patch16_rope_reg1_gap_256
vit_base_patch16_rpn_224
vit_base_patch16_siglip_224
vit_base_patch16_siglip_256
vit_base_patch16_siglip_384
vit_base_patch16_siglip_512
vit_base_patch16_siglip_gap_224
vit_base_patch16_siglip_gap_256
vit_base_patch16_siglip_gap_384
vit_base_patch16_siglip_gap_512
vit_base_patch16_xp_224
vit_base_patch32_224
vit_base_patch32_384
vit_base_patch32_clip_224
vit_base_patch32_clip_256
vit_base_patch32_clip_384
vit_base_patch32_clip_448
vit_base_patch32_clip_quickgelu_224
vit_base_patch32_plus_256
vit_base_r26_s32_224
vit_base_r50_s16_224
vit_base_r50_s16_384
vit_base_resnet26d_224
vit_base_resnet50d_224
vit_betwixt_patch16_gap_256
vit_betwixt_patch16_reg1_gap_256
vit_betwixt_patch16_reg4_gap_256
vit_betwixt_patch16_reg4_gap_384
vit_betwixt_patch16_rope_reg4_gap_256
vit_betwixt_patch32_clip_224
vit_giant_patch14_224
vit_giant_patch14_clip_224
vit_giant_patch14_dinov2
vit_giant_patch14_reg4_dinov2
vit_giant_patch16_gap_224
vit_gigantic_patch14_224
vit_gigantic_patch14_clip_224
vit_gigantic_patch14_clip_quickgelu_224
vit_huge_patch14_224
vit_huge_patch14_clip_224
vit_huge_patch14_clip_336
vit_huge_patch14_clip_378
vit_huge_patch14_clip_quickgelu_224
vit_huge_patch14_clip_quickgelu_378
vit_huge_patch14_gap_224
vit_huge_patch14_xp_224
vit_huge_patch16_gap_448
vit_intern300m_patch14_448
vit_large_patch14_224
vit_large_patch14_clip_224
vit_large_patch14_clip_336
vit_large_patch14_clip_quickgelu_224
vit_large_patch14_clip_quickgelu_336
vit_large_patch14_dinov2
vit_large_patch14_reg4_dinov2
vit_large_patch14_xp_224
vit_large_patch16_224
vit_large_patch16_384
vit_large_patch16_siglip_256
vit_large_patch16_siglip_384
vit_large_patch16_siglip_gap_256
vit_large_patch16_siglip_gap_384
vit_large_patch32_224
vit_large_patch32_384
vit_large_r50_s32_224
vit_large_r50_s32_384
vit_little_patch16_reg1_gap_256
vit_little_patch16_reg4_gap_256
vit_medium_patch16_clip_224
vit_medium_patch16_gap_240
vit_medium_patch16_gap_256
vit_medium_patch16_gap_384
vit_medium_patch16_reg1_gap_256
vit_medium_patch16_reg4_gap_256
vit_medium_patch16_rope_reg1_gap_256
vit_medium_patch32_clip_224
vit_mediumd_patch16_reg4_gap_256
vit_mediumd_patch16_reg4_gap_384
vit_mediumd_patch16_rope_reg1_gap_256
vit_pwee_patch16_reg1_gap_256
vit_relpos_base_patch16_224
vit_relpos_base_patch16_cls_224
vit_relpos_base_patch16_clsgap_224
vit_relpos_base_patch16_plus_240
vit_relpos_base_patch16_rpn_224
vit_relpos_base_patch32_plus_rpn_256
vit_relpos_medium_patch16_224
vit_relpos_medium_patch16_cls_224
vit_relpos_medium_patch16_rpn_224
vit_relpos_small_patch16_224
vit_relpos_small_patch16_rpn_224
vit_small_patch8_224
vit_small_patch14_dinov2
vit_small_patch14_reg4_dinov2
vit_small_patch16_18x2_224
vit_small_patch16_36x1_224
vit_small_patch16_224
vit_small_patch16_384
vit_small_patch32_224
vit_small_patch32_384
vit_small_r26_s32_224
vit_small_r26_s32_384
vit_small_resnet26d_224
vit_small_resnet50d_s16_224
vit_so150m2_patch16_reg1_gap_256
vit_so150m_patch16_reg4_gap_256
vit_so150m_patch16_reg4_gap_384
vit_so150m_patch16_reg4_map_256
vit_so400m_patch14_siglip_224
vit_so400m_patch14_siglip_378
vit_so400m_patch14_siglip_384
vit_so400m_patch14_siglip_gap_224
vit_so400m_patch14_siglip_gap_378
vit_so400m_patch14_siglip_gap_384
vit_so400m_patch14_siglip_gap_448
vit_so400m_patch14_siglip_gap_896
vit_so400m_patch16_siglip_256
vit_so400m_patch16_siglip_gap_256
vit_srelpos_medium_patch16_224
vit_srelpos_small_patch16_224
vit_tiny_patch16_224
vit_tiny_patch16_384
vit_tiny_r_s16_p8_224
vit_tiny_r_s16_p8_384
vit_wee_patch16_reg1_gap_256
vit_xsmall_patch16_clip_224

deit_base_distilled_patch16_224
deit_base_distilled_patch16_384
deit_base_patch16_224
deit_base_patch16_384
deit_small_distilled_patch16_224
deit_small_patch16_224
deit_tiny_distilled_patch16_224
deit_tiny_patch16_224
"""