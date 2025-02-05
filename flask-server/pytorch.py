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

pytorch_bp = Blueprint('pytorch', __name__)

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
    df = pd.read_csv(csv_file)
    return dict(zip(df['index'], df['class_name']))  

@pytorch_bp.route('/heatmap', methods=['POST'])
def heatmap():
    model_name = request.form.get('model_name')
    target_layer = request.form.get('target_layer')
    weights_file = request.files.get('weights_path') 
    image_file = request.files['image_path']
    class_labels_file = request.files['class_labels_csv']  

    if not model_name or not target_layer:
        return jsonify({"error": "Model name and target layer are required"}), 400

    class_labels = load_class_labels(class_labels_file)

    # load model
    if weights_file:
        print(f"Loading {model_name} with custom weights...")
        model = getattr(models, model_name)(weights=None)
        state_dict = torch.load(io.BytesIO(weights_file.read()), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Loading {model_name} with PyTorch's pretrained weights...")
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")


    if target_layer not in dict(model.named_modules()):
        return jsonify({"error": f"Invalid target layer '{target_layer}'"}), 400

    wrapped_model = CustomModelWrapper(model, target_layer)
    wrapped_model.eval()

    # image transformation
    input_sizes = {"inception_v3": (299, 299)}
    resize_dim = input_sizes.get(model_name, (224, 224))

    transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)


    output = wrapped_model(input_tensor)
    probabilities = F.softmax(output, dim=1).detach().cpu().numpy()

    # predicted class
    predicted_class_idx = output.argmax(dim=1).item()
    predicted_class_name = class_labels.get(predicted_class_idx, f"Unknown ({predicted_class_idx})")
    predicted_class_prob = probabilities[0, predicted_class_idx] * 100

    print(f"Predicted Class: {predicted_class_name} ({predicted_class_prob:.2f}%)")

    wrapped_model.zero_grad()
    output[0, predicted_class_idx].backward()

    gradients = wrapped_model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = wrapped_model.get_activations(input_tensor).detach()

    # Grad-CAM
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)

    # heatmap
    heatmap /= np.max(heatmap)
    heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    original_img = np.array(img)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colormap, 0.5, 0)

    # image to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    response = {
        "image": image_base64,
        "predicted_class": predicted_class_name,
        "predicted_probability": f"{predicted_class_prob:.2f}%"
    }

    return jsonify(response), 200


# Acceptabble models include: 
# resnet18, resnet34, resnet50, resnet101, resnet152 --> with target layer4
# alexnet --> with target features.11 
# vgg11, vgg13 --> with target features.20 
# vgg16, vgg19 --> with target features.29 
# inception_v3 --> with target Mixed_7c 
# densenet121, densenet169, densenet201, densenet161 --> with target features.norm5
# mobilenet_v2 --> with target features.18  
# efficientnet_b0, efficientnet_b1, ..., efficientnet_b7 --> with target features.7 
# squeezenet1_0, squeezenet1_1 --> with target features.12  --> NOT
# shufflenet_v2_x0_5, shufflenet_v2_x1_0 --> with target conv5
# mnasnet0_5, mnasnet1_0 --> with target layers[-1] --> NOT 