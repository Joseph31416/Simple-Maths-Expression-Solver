import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def train_label_retrieval2():
    """
    Returns train_label which is a (100000, ) np.array, containing label of each image as a string;
    """
    operation = {'plus': '+', 'minus': '-', 'times': '*', 'divide': '/'}
    with open("/content/gdrive/MyDrive/Colab_Notebooks/CRNN_Augmented/data_library/train.csv", "r") as f:
        header_list = f.readline().strip().split(',')
        train_label = []
        for line in f:
            working_list = line.strip().split(',')
            working_dict = {}
            for i, header in enumerate(header_list):
                working_dict[header] = working_list[i]
            cat = working_dict["num1"] + operation[working_dict["op"]] + working_dict["num2"]
            train_label.append(cat)
        return train_label


def image_preprocessing(images,
                        labels,
                        char_to_num,
                        batch_size=64,
                        train_size=0.8):
    """
    images:  (50000, 200, 50, 3) numpy array
    labels: (50000, ) numpy array
    char_to_num: tensorflow object
    """
    n = int(len(images) * train_size)
    x_train = images[:n] / 255
    y_train = labels[:n]
    x_valid = images[n:] / 255
    y_valid = labels[n:]
    del images
    del labels

    def encode_single_sample(img, label):
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    del x_train
    del y_train

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    del x_valid
    del y_valid

    return train_dataset, validation_dataset


def train_data_retrieval(a, b):
    """
    :param a: index of first file (included)
    :param b: index of last file (excluded)
    :return: train_dataset, validation_dataset
    """
    images = np.array([[[[0]] * 50] * 200])
    for i in range(a, b):
        data_dir = f"/content/gdrive/MyDrive/Colab_Notebooks/CRNN_Augmented/data_library/numpyies_aug/Train{i}.npy"
        images = np.append(images, np.load(data_dir), axis=0)
        # with np.load(data_dir) as data:
        #     images = np.append(images, data["arr_0"], axis=0)
    images = np.delete(images, 0, axis=0)
    labels = train_label_retrieval2()[(a-1)*10000:(b-1)*10000]
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

    print("Number of images found: ", len(images))
    print("Number of labels found: ", len(labels))
    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)

    # Mapping characters to integers
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary=characters, num_oov_indices=0, mask_token=None
    )

    train_dataset, validation_dataset = image_preprocessing(images, labels, char_to_num)
    del images, labels

    return train_dataset, validation_dataset


#Used only for testing during development
def dataset_visualisation(train_dataset):
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

    # Mapping characters to integers
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary=characters, num_oov_indices=0, mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    _, ax = plt.subplots(4, 4, figsize=(10, 5))
    for batch in train_dataset.take(1):
        images = batch["image"]
        labels = batch["label"]
        for i in range(16):
            img = (images[i] * 255).numpy().astype("uint8")
            label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
            ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")
    plt.show()
