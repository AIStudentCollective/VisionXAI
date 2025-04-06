import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import json

def load_class_mapping(csv_file):
    try:
        df = pd.read_csv(csv_file)
        class_mapping = {row['index']: row['class'] for _, row in df.iterrows()}
        return class_mapping
    except FileNotFoundError:
        error_message = f"Error: Class mapping file not found: {csv_file}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        error_message = f"Error: Class mapping file is empty: {csv_file}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError:
        error_message = f"Error: Could not parse class mapping file: {csv_file}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        error_message = f"Error: Missing column ('index' or 'class') in class mapping file: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_message = f"Error loading or processing class mapping file: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)


def load_and_preprocess_image(img_path, preprocess_input, target_size):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except FileNotFoundError:
        error_message = f"Error: Image file not found: {img_path}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_message = f"Error loading or preprocessing image: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    error_message = "No convolutional layer found in the model"
    print(json.dumps({"error": error_message}), file=sys.stderr)
    sys.exit(1)

def grad_cam(model, weights_path, img_path, preprocess_input):
    try:
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
        if original_img is None:
            error_message = f"Could not read image file: {img_path}"
            print(json.dumps({"error": error_message}), file=sys.stderr)
            sys.exit(1)

        original_img = cv2.resize(original_img, input_size) # Ensure the original image matches the heatmap size
        overlayed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

        pred_class_idx = tf.argmax(predictions[0]).numpy()

        # Use absolute path for output
        output_image_path = os.path.join(os.getcwd(), 'uploads', 'visualization.png')
        cv2.imwrite(output_image_path, overlayed_img)
        return pred_class_idx, output_image_path

    except Exception as e:
        error_message = f"Error during Grad-CAM processing: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run Grad-CAM on an image using a specified model architecture.')
    parser.add_argument('csv_file', type=str, help='CSV file with class mappings')
    parser.add_argument('weights_path', type=str, help='Path to the model weights (.h5 file)')
    parser.add_argument('architecture', type=str, help='Model architecture type (ResNet50/VGG16/InceptionV3/MobileNetV2)')
    parser.add_argument('image_path', type=str, help='Path to the image for inference')

    args = parser.parse_args()

    try:
        class_mapping = load_class_mapping(args.csv_file)

        if args.architecture == 'ResNet50':
            from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
            model = ResNet50(weights=None)
            preprocess = resnet_preprocess
        elif args.architecture == 'VGG16':
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
            model = VGG16(weights=None)
            preprocess = vgg_preprocess
        elif args.architecture == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
            model = InceptionV3(weights=None)
            preprocess = inception_preprocess
        else:
            raise ValueError("Unsupported architecture type.")

        pred_index, visualization_path = grad_cam(model=model,
                                weights_path=args.weights_path,
                                img_path=args.image_path,
                                preprocess_input=preprocess)

        predicted_class_name = class_mapping.get(pred_index)

        result = {
            "predicted_class_name": predicted_class_name,
            "visualization_path": visualization_path
        }

        print(json.dumps(result)) # Print JSON to standard output

    except Exception as e:
        error_message = f"An error occurred in main: {e}"
        print(json.dumps({"error": error_message}), file=sys.stderr)  # JSON error to stderr
        sys.exit(1)

if __name__ == '__main__':
    main()
