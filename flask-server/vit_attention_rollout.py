import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import json
import pandas as pd
import io
import base64
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor

# For HF ViTs
def loadHFModel(modelName, weightsFilePath=None, num_classes=2):
    model = AutoModelForImageClassification.from_pretrained(modelName)
    state_dict = torch.load(weightsFilePath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(modelName)
    return model, feature_extractor

def computeRollout(model, feature_extractor, image_path, labels, num_classes):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    inputs = feature_extractor(images=image, return_tensors="pt")


#---- For Pytorch ViTs -----
def loadModel(modelName, weightsFilePath, numClasses=2):
    # Load model
    try:
        model = models.get_model(modelName, weights=None)
        print(f'Loaded model for {modelName} from Pytorch pretrained models.')
    except Exception as e:
        raise ValueError(f'Error loading model: {e}')

    # Adjust classification according to labels
    try:
        numClassesLocal = numClasses
        if (numClassesLocal != 1000):
            # Replace classification head
            if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                inFeatures = model.heads.head.in_features
                model.heads.head = nn.Linear(inFeatures, numClassesLocal)
                print(f"Replaced classifier head to output {numClassesLocal} classes.")
            # Replace classifier head (for older ViT models)
            elif hasattr(model, 'classifier'):
                inFeatures = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
                model.classifier = nn.Linear(inFeatures, numClassesLocal)
                print(f"Replaced classifier layer to output {numClassesLocal} classes (using 'classifier' attribute).")
    except Exception as e:
        raise ValueError(f'Error adjusting classification head: {e}')
    # Load weights
    try:
        model.load_state_dict(torch.load(weightsFilePath, weights_only=True, map_location=torch.device('cpu')))
        print(f'Loaded weights from {weightsFilePath}')
    except Exception as e:
        raise ValueError(f'Error in loading weights: {e}')

    model.to('cpu').eval()
    return model

def loadCsv(csvLabelFilePath):
    try:
        labelsDf = pd.read_csv(csvLabelFilePath)
        labels = labelsDf.set_index('index')['label'].to_dict()
        return labels
    except Exception as e:
        raise ValueError(f'Error in loading csv labels file: {e}')

def preprocessImage(model, imageFilePath):
    try:
        inputSize = tuple((224, 224))

        transformPipeline = transforms.Compose([
            transforms.Resize(inputSize, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        img = Image.open(imageFilePath).convert('RGB')
        imgTensor = transformPipeline(img).unsqueeze(0)
        return imgTensor, img
    except Exception as e:
        raise ValueError(f'Error in preprocessing image: {e}')

def computeRollout(attentions, discardRatio=0.9):
    numLayers = len(attentions)
    batchSize, numHeads, seqLen, _ = attentions[0].shape
    identity = torch.eye(seqLen).to(attentions[0].device)
    for layerAttn in attentions:
        layerAttn = layerAttn.clone()
        if discardRatio > 0:
            _, sortedIndices = torch.sort(layerAttn.mean(dim=(-2, -1)))
            discardNum = int(discardRatio * numHeads)
            for i in range(discardNum):
                layerAttn[:, sortedIndices[:, i], :, :] = 0
        layerAttn = layerAttn / layerAttn.sum(dim=-1, keepdim=True)
        resAttn = layerAttn @ identity
        identity = resAttn
    rollout = identity.mean(dim=1)
    return rollout

def visualizeRollout(rolloutMap, originalImagePath, outputPath="attention_rollout_vit.png", patchSize=16): # Same visualization
    img = Image.open(originalImagePath).convert('RGB')
    imgSize = img.size[0]
    rolloutMapResized = torch.nn.functional.interpolate(
        rolloutMap.unsqueeze(0).unsqueeze(0),
        size=(imgSize, imgSize),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    rolloutMapNormalized = (rolloutMapResized - rolloutMapResized.min()) / (rolloutMapResized.max() - rolloutMapResized.min())
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, alpha=1.0)
    ax.imshow(rolloutMapNormalized, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.title("Attention Rollout Visualization")

    imgBuf = io.BytesIO()
    plt.savefig(imgBuf, format='png', bbox_inches='tight')
    imgBuf.seek(0)
    imageBase64 = base64.b64encode(imgBuf.read()).decode('utf-8')

    plt.close(fig)

    return imageBase64

def vitAttHook(module, input, output, attentionMatricesList):
    attnWeights = output[1]
    if attnWeights is not None:
        attentionMatricesList.append(attnWeights)
    else:
        print(f"Attention weights are None in {module.__class__.__name__}. Skipping.")

def registerAttHooks(model, attentionMatricesList):
    hookedLayerCount = 0
    for name, module in model.named_modules():
        if isinstance(module, models.vision_transformer.VisionTransformerEncoder):
            for blockIndex, block in enumerate(module.blocks):
                if hasattr(block, 'attn') and isinstance(block.attn, nn.MultiheadAttention): # Check for attention layer in block
                    attentionLayer = block.attn
                    print(f"Registering hook on block {blockIndex}, Attention layer: {attentionLayer.__class__.__name__}")
                    attentionLayer.register_forward_hook(lambda module, input, output: vitAttHook(module, input, output, attentionMatricesList)) # Using lambda to pass attentionMatricesList
                    hookedLayerCount += 1
    if hookedLayerCount == 0:
        print("Warning: No MultiheadAttention layers found in the ViT model structure. Attention rollout may not be possible.")
    else:
        print(f"Registered forward hooks on {hookedLayerCount} MultiheadAttention layers.")


