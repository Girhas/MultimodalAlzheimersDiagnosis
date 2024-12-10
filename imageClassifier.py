import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Multiply
import tempfile 
import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

@tf.keras.utils.register_keras_serializable(package='Custom')
class SpatialAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):  # Accept additional keyword arguments
        super(SpatialAttentionLayer, self).__init__(**kwargs)  # Pass kwargs to the parent class
        # Initialize the Conv2D layer only once in the __init__ method
        self.conv2d = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, input_feature):
        # Compute spatial attention
        avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # Apply the Conv2D layer defined in __init__
        attention = self.conv2d(concat)

        # Multiply attention weights with the input feature map
        return layers.Multiply(name='attention_multiply')([input_feature, attention])

    @classmethod
    def from_config(cls, config):
        return cls(**config)  # Ensure to pass config to the constructor
def plot_attention_map(model, image):
    # Get the output from the attention layer (assuming the layer's name is 'attention_multiply')
    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer('spatial_attention_layer').output)

    # Use predict to get actual data, not symbolic tensor
    attention_map = intermediate_layer_model.predict(image)

    # Average the attention map across the channels to reduce it to a 2D map
    attention_map = np.mean(attention_map, axis=-1)  # Reduces shape to (160, 160)

    # Resize the attention map to the size of the input image
    attention_map_resized = tf.image.resize(attention_map[..., np.newaxis], (image.shape[1], image.shape[2])).numpy()  # Now (160, 160)

    # Remove the batch dimension and ensure the shape is correct for matplotlib
    attention_map_resized = np.squeeze(attention_map_resized, axis=0)  # Shape: (160, 160)

    # Normalize the attention map for better visualization
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())

    # Convert the image to a displayable format (assuming the original image is in RGB format)
    image_for_display = np.squeeze(image, axis=0)  # Shape: (160, 160, 3) - original input image should be RGB

    # Plot the original image with attention overlay
    plt.imshow(image_for_display)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.4)  # Overlay the attention map
    plt.title('Attention Map Overlay')
    plt.colorbar()

    # Save the plot to a temporary file and return the file path
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name
best_model = load_model('models/best_overall_model_NASNetLarge_atten.keras')

def draw_attention_path (image):
    processed_image = preprocess_image(image)
    return plot_attention_map(best_model, processed_image)
# Function to preprocess the input image
def preprocess_image(image_path, target_size=(160, 160)):  # Adjust target_size based on your model input
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  # Rescale pixel values if necessary (based on model training)
    return img_array
def predict_image(image_path):
    # Path to the single image file
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make a prediction
    predictions = best_model.predict(processed_image)



    # Get the predicted class and the confidence score (probability)
    predicted_class = np.argmax(predictions, axis=1)
    confidence_score = np.max(predictions, axis=1)  # Confidence is the max probability
    # Define your label map
    label_map = {0: "AD", 1: "CN"}
    predicted_class = np.array([0])  
    predicted_class = predicted_class.item()  
    predicted_label = label_map[predicted_class]
    return predicted_label, confidence_score