import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os

def load_class_mapping(csv_file):
	df= pd.read_csv(csv_file) 
	class_mapping= {row['index']: row['class'] for _, row in df.iterrows()}  
	return class_mapping 

def load_and_preprocess_image(img_path, preprocess_input, target_size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No convolutional layer found in the model")

def grad_cam(model, weights_path, img_path, preprocess_input):
    model.load_weights(weights_path)
    model.trainable = False  # Important: freeze the model weights
    input_size = model.input_shape[1:3] 

    img_array = load_and_preprocess_image(img_path, preprocess_input, target_size=input_size)
    last_conv_layer = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])] # Get the loss for the predicted class

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    #  Convert heatmap to NumPy array *before* any further operations:
    heatmap = heatmap.numpy()  # Crucial correction

    heatmap = np.maximum(heatmap.squeeze(), 0) / np.max(heatmap)  # Normalize (also simplified)
    heatmap = cv2.resize(heatmap, input_size) # Now heatmap is guaranteed to be a numpy array
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, input_size) # Ensure the original image matches the heatmap size
    overlayed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    pred_class_idx = tf.argmax(predictions[0]).numpy()
    
    output_image_path = 'uploads/visualization.png'
    cv2.imwrite(output_image_path, overlayed_img)

    return pred_class_idx, output_image_path


def main():
   parser = argparse.ArgumentParser(description='Run Grad-CAM on an image using a specified model architecture.')
   parser.add_argument('csv_file', type=str, help='CSV file with class mappings')
   parser.add_argument('weights_path', type=str, help='Path to the model weights (.h5 file)')
   parser.add_argument('architecture', type=str, help='Model architecture type (ResNet50/VGG16/InceptionV3/MobileNetV2)')
   parser.add_argument('image_path', type=str, help='Path to the image for inference')

   args = parser.parse_args()

   class_mapping = load_class_mapping(args.csv_file)

   if args.architecture == 'ResNet50':
       from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
       model = ResNet50(weights=None)
   elif args.architecture == 'VGG16':
       from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
       model = VGG16(weights=None)
   elif args.architecture == 'InceptionV3':
       from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
       model = InceptionV3(weights=None)
   else:
       raise ValueError("Unsupported architecture type.")

   pred_index, visualization_path = grad_cam(model=model,
                          weights_path=args.weights_path,
                          img_path=args.image_path,
                          preprocess_input=resnet_preprocess if args.architecture == 'ResNet50' else 
                                           vgg_preprocess if args.architecture == 'VGG16' else 
                                           inception_preprocess)

   predicted_class_name = class_mapping.get(pred_index)

   print(f"{predicted_class_name},{visualization_path}")

if __name__ == '__main__':
   main()
