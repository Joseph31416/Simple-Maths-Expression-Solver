from tensorflow import keras


def train_model(
        model,
        train_dataset,
        validation_dataset,
        epochs=100,
        patience=10,
        **kwargs
        ):
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    restore_best_weights=True)]
    new = kwargs.get('new', True)
    saved_model_dir = kwargs.get('saved_model_dir',
                                 "/content/gdrive/MyDrive/Colab_Notebooks/CRNN_Augmented/saved_models/predictions_CRNN_Aug1.h5")
    load_model_dir = kwargs.get('load_model_dir', saved_model_dir)
    
    if not new:
        model.load_weights(load_model_dir)
    else:
        model.save(saved_model_dir)

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        shuffle=True
    )

    model.save(saved_model_dir)
