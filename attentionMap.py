class SpatialAttentionLayer(keras.layers.Layer):
    def __init__(self):
        super(SpatialAttentionLayer, self).__init__()
        # Initialize the Conv2D layer only once in the __init__ method
        self.conv2d = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, input_feature):
        # Compute spatial attention
        avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # Apply the Conv2D layer defined in __init__
        attention = self.conv2d(concat)

        # Multiply attention weights with the input feature map
        return Multiply(name='attention_multiply')([input_feature, attention])

# Function to build the model with attention

def build_model(hp):
    base_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    x = base_model.output

    # Add attention before final layers
    x = SpatialAttentionLayer()(x)

    # Global average pooling and final dense layers
    x = GlobalAveragePooling2D()(x)

    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    x = Dropout(rate=dropout_rate)(x)

    units = hp.Int('units', min_value=64, max_value=256, step=64)
    x = Dense(units, activation='relu')(x)

    skip = Dense(units, activation='relu')(x)
    x = Add()([x, skip])

    x = Dropout(rate=dropout_rate)(x)

    predictions = Dense(len(class_names), activation='softmax')(x)
    model = keras.Model(inputs=base_model.input, outputs=predictions)

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


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

    # Display the original image
    plt.imshow(image_for_display)

    # Overlay the attention map on the original image with transparency
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.4)  # Set alpha lower so the original image is visible

    plt.title('Attention Map Overlay')
    plt.colorbar()
    plt.show()

