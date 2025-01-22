from flask import Blueprint, request, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io

pytorch_bp = Blueprint('pytorch', __name__)

class CustomModelWrapper(nn.Module):
    def __init__(self, model, target_layer_name):
        super(CustomModelWrapper, self).__init__()
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        target_layer = dict(self.model.named_modules())[target_layer_name]
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


@pytorch_bp.route('/heatmap', methods=['POST'])
def heatmap():
    model_name = request.form.get('model_name')
    target_layer = request.form.get('target_layer')
    weights_file = request.files.get('weights_path')
    image_file = request.files.get('image_path')

    # Load weights and image from memory
    weights_bytes = io.BytesIO(weights_file.read())
    image_bytes = io.BytesIO(image_file.read())

    input_sizes = {"inception_v3": (299, 299)}
    resize_dim = input_sizes.get(model_name, (224, 224))

    # Initialize model
    model = getattr(models, model_name)(weights=None)
    if hasattr(model, "classifier"):
        model.classifier = nn.Linear(model.classifier.in_features, 2)  # Binary classification
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, 2)

    # Load the weights
    state_dict = torch.load(weights_bytes, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    wrapped_model = CustomModelWrapper(model, target_layer)
    wrapped_model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_bytes).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    # Forward pass
    output = wrapped_model(input_tensor)
    probabilities = F.softmax(output, dim=1).detach().cpu().numpy()
    pneumonia_positive_percentage = probabilities[0, 1] * 100
    pneumonia_negative_percentage = probabilities[0, 0] * 100
    predicted_class = output.argmax(dim=1).item()
    print(f"Pneumonia Positive: {pneumonia_positive_percentage:.2f}%")
    print(f"Pneumonia Negative: {pneumonia_negative_percentage:.2f}%")
    print(f"Predicted Class: {predicted_class} (0 = No Pneumonia, 1 = Pneumonia)")

    # Backward pass
    wrapped_model.zero_grad()
    output[0, predicted_class].backward()

    gradients = wrapped_model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = wrapped_model.get_activations(input_tensor).detach()

    # Grad-CAM heatmap generation
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)

    # Normalize heatmap
    threshold = np.percentile(heatmap, 70)
    heatmap[heatmap < threshold] = 0
    heatmap /= np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Overlay heatmap on the original image
    original_img = np.array(img)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colormap, 0.5, 0)

    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    result_bytes = io.BytesIO(buffer)

    return send_file(result_bytes, mimetype='image/png')
