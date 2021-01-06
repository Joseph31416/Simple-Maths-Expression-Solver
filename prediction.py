import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def test_image_preprocessing(images,
                             labels,
                             char_to_num,
                             batch_size=128):
    """
    Converts numpy array to tensorflow object
    images:  (20000, 200, 50, 3) numpy array
    labels: (20000, ) numpy array
    char_to_num: tensorflow object
    returns test_dataset
    """
    x_test = images
    y_test = labels
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

    # Mapping characters to integers
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary=characters, num_oov_indices=0, mask_token=None
    )

    def encode_single_sample(img, label):
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = (
        test_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return test_dataset


def test_data_generator(image, char_to_num):

    test_images = image
    test_labels = ['1-3']*len(test_images)

    # Mapping characters to integers
    test_dataset = test_image_preprocessing(test_images, test_labels, char_to_num)
    del test_images
    return test_dataset


def get_prediction_model(model):
    """
    Retrieve layers involve in prediction from the network
    """
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    return prediction_model


def decode_batch_predictions(pred, num_to_char):
    """
    Takes in predictions (pred) and converts it to desired strings
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :3
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def rounding2dp(n):
    """
    Round up at half for positive numbers, round down at half for negative numbers
    """
    n_str = str(n)
    post_dec = n_str.strip().split('.')[1]
    if len(post_dec) <= 2:
        return n
    else:
        if post_dec[2] == '5':
            n = ((n * 100) // 1 + 1) / 100
        else:
            n = round(n, 2)
        return n


def calculation(cat_tuple):
    num = '0123456789'
    op = '+-*/'
    if cat_tuple[0] in num and cat_tuple[1] in op and cat_tuple[2] in num:
        if cat_tuple[1] == "+":
            return str(int(cat_tuple[0]) + int(cat_tuple[2])) + '.00'
        elif cat_tuple[1] == "-":
            return str(int(cat_tuple[0]) - int(cat_tuple[2])) + '.00'
        elif cat_tuple[1] == "*":
            return str(int(cat_tuple[0]) * int(cat_tuple[2])) + '.00'
        else:
            if cat_tuple[2] == '0':
                value = str(rounding2dp(int(cat_tuple[0]) / 6))
            else:
                value = str(rounding2dp(int(cat_tuple[0]) / int(cat_tuple[2])))
            if len(value.split('.')[1]) == 1:
                value = value + '0'
            return value
    else:
        return 'UKN'

# def result_visualisation(test_dataset):
#     for batch in test_dataset.take(1):
#         image = batch["image"]
#         img = (image[0, :, :, 0] * 255).numpy().astype(np.uint8)
#         img = img.T
#         plt.imshow(img, cmap="gray")
#     plt.show()


def get_predictions(model, image):
    value = None
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

    # Mapping characters to integers
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary=characters, num_oov_indices=0, mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    prediction_model = get_prediction_model(model)
    test_dataset = test_data_generator(image, char_to_num)
    # result_visualisation(test_dataset)
    for batch in test_dataset.take(1):
        pred = prediction_model.predict(batch["image"])
        pred_text = decode_batch_predictions(pred, num_to_char)
        value = calculation(pred_text[0])
        value = str(value)
    return value
