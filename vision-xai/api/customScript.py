import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
import sys


def get_target_size(model):
    input_shape = model.input_shape
    return (input_shape[1], input_shape[2])

def load_and_preprocess_image(img_path, model):
    target_size = get_target_size(model)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:  # Check if it's a convolutional layer (4D output)
            return layer.name
    raise ValueError("Could not find a convolutional layer.")  # Raise error if no convolutional layer is found

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]  # Get the output of the last convolutional layer
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]  # Calculate the heatmap
    heatmap = tf.squeeze(heatmap)  # Remove dimensions of size 1

    # Normalize heatmap to 0-1 range
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # Use tf.math.reduce_max
    return heatmap.numpy() # Convert to NumPy array



def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    heatmap = np.uint8(255 * heatmap)  # Convert to uint8 BEFORE resizing
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img.astype(np.uint8), 0.6, heatmap_colored.astype(np.uint8), 0.4, 0)
    return superimposed_img


def gradcam(model_path, img_path, csv_file):  # Added csv_file argument
    try:
        print(model_path)
        model = tf.keras.models.load_model(model_path)
    except Exception as e: # Catch file loading errors
        print(f"Error loading model: {e}", file=sys.stderr)
        raise

    img_array = load_and_preprocess_image(img_path, model)

    try:
        last_conv_layer_name = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        superimposed_img = overlay_heatmap(img_path, heatmap)
    except ValueError as e:  # Catch convolutional layer or heatmap errors
        print(f"Error generating heatmap: {e}", file=sys.stderr)
        raise
    except Exception as e:  # Catch other image processing errors
        print(f"An unexpected error occurred during image processing: {e}", file=sys.stderr)
        raise


    visualization_filename = 'uploads/visualization.png'  # Save visualization in uploads directory
    output_image_path = os.path.abspath(visualization_filename)  # Get absolute path

    try:
        cv2.imwrite(output_image_path, superimposed_img)
    except Exception as e: # Catch errors when saving image
        print(f"Error saving visualization: {e}", file=sys.stderr)
        raise


    pred_index = np.argmax(model.predict(img_array))


    class_mapping = load_class_mapping(csv_file)
    predicted_class_name = class_mapping.get(pred_index, "Unknown") # Added error handling for get


    # Return the predicted class name and the path (consistent with gcamScript.py)
    return predicted_class_name, output_image_path



def load_class_mapping(csv_file):
	try:
		df= pd.read_csv(csv_file)
		class_mapping = {row['index']: row['class'] for _, row in df.iterrows()}
		return class_mapping
	except FileNotFoundError:
		print(f"Error: Class mapping file not found: {csv_file}", file=sys.stderr)
		raise
	except pd.errors.EmptyDataError:
		print(f"Error: Class mapping file is empty: {csv_file}", file=sys.stderr)
		raise
	except pd.errors.ParserError:
		print(f"Error: Could not parse class mapping file: {csv_file}", file=sys.stderr)
		raise
	except KeyError as e:
		print(f"Error: Missing column ('index' or 'class') in class mapping file: {e}", file=sys.stderr)
		raise

def main():
	parser= argparse.ArgumentParser(description='Run Grad-CAM on an image using a custom architecture.')
	parser.add_argument('csv_file', type=str ,help='CSV file with class mappings')
	parser.add_argument('model_path', type=str ,help='Path to the custom model (.h5 file)')
	parser.add_argument('image_path', type=str ,help='Path to the image for inference')

	args= parser.parse_args()

	class_mapping= load_class_mapping(args.csv_file)

	pred_index , visualization_path= gradcam(args.model_path, args.image_path, args.csv_file)

	predicted_class_name= class_mapping.get(pred_index)

	print(f"{predicted_class_name},{visualization_path}")

if __name__== '__main__':
	main()
