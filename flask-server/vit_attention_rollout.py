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
import cv2
import seaborn as sns
from tqdm.auto import tqdm
import timm
# from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def readConfig(config_file=None):
    if config_file is None:
        return None
    print('Reading config file.')
    with open(config_file) as file:
        data = json.load(file)
    return data

def preprocessImage(image, image_size=224):
    print('Preprocessing image.')
    img = Image.open(image).convert('RGB')
    transform = transforms.Compose([transforms.Resize(249, 3), 
                            transforms.CenterCrop(image_size), 
                            transforms.ToTensor(), 
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(img)

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 1:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

def attention_rollout_function(attn_maps):
    attn_rollout = []
    I = torch.eye(attn_maps[0].shape[-1])  # Identity matrix
    prod = I
    for i, attn_map in enumerate(attn_maps):
        prod = prod @ (attn_map + I)  # Product of attention maps with identity matrix
        
        prod = prod / prod.sum(dim=-1, keepdim=True) # Normalize
        attn_rollout.append(prod)
    return attn_rollout

def loadModel(model_name, weights_file):
    print(f"Creating {model_name} from TIMM")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = timm.create_model(model_name).to(device)
    # return Exception(f'Model {model_name} unsupported. Only Facebook\'s DeiTs and Google\'s ViTs are supported right now.')
    print('Model created.')
    if weights_file:
        print(f'Loading weights from {weights_file}')
        model.load_state_dict(torch.load(weights_file, weights_only=True, map_location=torch.device(device)))
        print('Weights loaded.')
    
    print('Setting model to eval mode.')
    model.eval()
    return model

def readCSV(class_labels_file):
    print('Read CSV to dictionary')
    df = pd.read_csv(class_labels_file)
    return dict(zip(df['index'], df['class_name']))

def makeRollout(model_name, weights_file, image_file, class_labels_file, num_attention_heads, image_size=224):
    print('Attempting to make attention rollout.')
    try:
        model = loadModel(model_name, weights_file)
        print('Model loaded.')
        
        patches = model.patch_embed.patch_size 
        print(f'Patches: {patches}')
        print(f'Image file: {image_file}')
        print(f'Image size: {image_size}')

        patch_size = patches[0]
        print(f'Patch size: {patch_size}')
        print(f'Patch size type: {type(patch_size)}')
        image_size = int(image_size)
        print(f'Image size type: {type(image_size)}')
        num_patches = int((int(image_size))/patch_size)
        print(f'Number of patches: {num_patches}')
        
        print(f'Class labels file: {class_labels_file}')
        class_labels = readCSV(class_labels_file)
        print('Class labels read.')
        
        data = readConfig()
        print('Config file read.')
        
        if data is None:
            num_attention_heads = 12
        else:
            num_attention_heads = data.get('num_attention_heads', 12)
        print(f'Number of attention heads: {num_attention_heads}')
        
        img_tensor = preprocessImage(image_file, image_size)
        print('Image preprocessed.')

        model.blocks[-1].attn.forward = my_forward_wrapper(model.blocks[-1].attn)
        y = model(img_tensor.unsqueeze(0).to(device))
        attn_map = model.blocks[-1].attn.attn_map.mean(dim=1).squeeze(0).detach()
        cls_weight = model.blocks[-1].attn.cls_attn_map.max(dim=1).values.view(num_patches, num_patches).detach()
        img_resized = img_tensor.permute(1, 2, 0) * 0.5 + 0.5
        cls_resized = F.interpolate(cls_weight.view(1, 1, num_patches, num_patches), (image_size, image_size), mode='bilinear').view(image_size, image_size, 1)
        attn_map_cpu = attn_map.cpu()
        cls_weight_cpu = cls_weight.cpu()

        print('Line 131 reached.')
        # model = loadModel(model_name, weights_file)
        for block in tqdm(model.blocks):
            block.attn.forward = my_forward_wrapper(block.attn)

        y = model(img_tensor.unsqueeze(0).to(device))
        attn_maps = []
        cls_weights = []
        for block in tqdm(model.blocks):
            attn_maps.append(block.attn.attn_map.max(dim=1).values.squeeze(0).detach())
            cls_weights.append(block.attn.cls_attn_map.mean(dim=1).view(num_patches, num_patches).detach())

        img_resized = img_tensor.permute(1, 2, 0) * 0.5 + 0.5
        print('Image resized.')

        attn_maps_cpu = []
        for i in range(num_attention_heads):
            attn_map = attn_maps[i]
            attn_map_cpu = attn_map.cpu()
            attn_maps_cpu.append(attn_map_cpu)

        cls_weights_cpu = []
        for i in range(num_attention_heads):
            cls_weight = cls_weights[i]
            cls_weight_cpu = cls_weight.cpu()
            cls_weights_cpu.append(cls_weight_cpu)

        attn_rollout = attention_rollout_function(attn_maps_cpu)
        cls_weights_rollout = []
        cls_resized_rollout = []

        for i in tqdm(range(num_attention_heads)):
            cls_weights_rollout.append(attn_rollout[i][0, 1:])
            cls_weights_rollout[i] = cls_weights_rollout[i].view(num_patches, num_patches)
            cls_resized_rollout.append(F.interpolate(cls_weights_rollout[i].view(1, 1, num_patches, num_patches), (image_size, image_size), mode='bilinear').view(image_size, image_size, 1))

        cls_weight = cls_weights_rollout[num_attention_heads-1]
        cls_resized = F.interpolate(cls_weight.view(1, 1, num_patches, num_patches), (image_size, image_size), mode='bilinear').view(image_size, image_size, 1)

        print('Line 170 reached.')
        cls_weight = cls_weights_rollout[11]
        cls_resized = F.interpolate(cls_weight.view(1, 1, num_patches, num_patches), (image_size, image_size), mode='bilinear').view(image_size, image_size, 1)

        plt.imshow(img_resized)
        plt.imshow(cls_resized, alpha=0.5)
        plt.axis('off')
        # plt.title('Attention Rollout Visualization')
                
        imgBuf = io.BytesIO()
        print('Saving image to buffer.')
        plt.savefig(imgBuf, format='png', bbox_inches='tight', pad_inches=0)
        imgBuf.seek(0)
        imageBase64 = base64.b64encode(imgBuf.read()).decode('utf-8')
        print('Image saved to buffer.')

        probabilities = F.softmax(y, dim=1).squeeze(0).cpu().detach().numpy()
        print('Probabilities calculated.')
        top_class = np.argmax(probabilities)
        print('Top class calculated.')
        # top_class_label = class_labels[str(top_class[0])]
        # print('Top class label:', top_class_label)
        top_class_prob = probabilities[top_class]
        print('Top class probability:', top_class_prob)
        # print('Top class probability:', top_class_prob)

        response = {
            'image': imageBase64,
            'predicted_class': str(top_class),
            'predicted_probability': str(top_class_prob * 100),
        }
        # print(f'Response: {response}')
        return response
    
    except Exception as e:
        return ValueError(f'Attention rollout error. {e}')