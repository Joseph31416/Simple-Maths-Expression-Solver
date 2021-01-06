import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model(img_width=200, img_height=50, lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1):
    # Inputs to the model
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(32, (3, 3), activation="selu", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="selu", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), activation="selu", padding="same", name="Conv3")(x)
    x = layers.Conv2D(128, (3, 3), activation="selu", padding="same", name="Conv4")(x)

    x = layers.MaxPooling2D((2, 1), name="pool3")(x)
    x = layers.Conv2D(256, (3, 3), activation="selu", padding="same", name="Conv5")(x)

    x = layers.BatchNormalization(name="bn_1")(x)

    x = layers.Conv2D(256, (3, 3), activation="selu", padding="same", name="Conv6")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling2D((2, 1), name="pool4")(x)

    x = layers.Conv2D(64, (2, 2), activation="selu", padding="same", name="Conv7")(x)
    x = layers.Dropout(0.1)(x)

    new_shape = ((img_width // 16), (img_height // 16) * 256)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(4096, activation="selu", name="dense7")(x)
    x = layers.Dense(128, activation="selu", name="dense8")(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, kernel_initializer="he_normal", return_sequences=True))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Bidirectional(layers.LSTM(128, kernel_initializer="he_normal", return_sequences=True))(x)
    # Output layer
    x = layers.Dense(len(characters) + 1, kernel_initializer="he_normal", activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, clipnorm=clipnorm)
    # Compile the model and return
    model.compile(optimizer=opt)
    return model
