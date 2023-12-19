import os
import sys
import logging
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from absl import app
from absl import flags


workspace_dir = os.environ.get('WORKSPACE_DIR', '/workspace')
path_lib_dir = os.path.join(workspace_dir, 'lib')
# check if path has been added
if os.path.dirname(os.path.abspath(path_lib_dir)) not in sys.path:
    sys.path.append(path_lib_dir)

from lib.dataset import Dataset
from lib.augment import Processing


def prepare_dataset(ds, seed, target_size=(224, 224), batch_size=64, augment=True, shuffle=True):
    preprocessing = Processing(
        random_flip_left_right=True,
        random_brightness=0.1,
        random_contrast=0.1,
        random_rotation=180,
        random_saturation=0.1,
        random_hue=0.05,
        random_crop=0.05,
        random_translate=0.05,
        random_zoom=0.05,
    )
    ds_prepared = preprocessing.prepare_for_training(
        ds=ds,
        seed=seed,
        target_size=target_size,
        batch_size=batch_size,
        augment=augment,
        shuffle=shuffle,
    )
    return ds_prepared


def create_model(
    model_class, 
    seed, 
    target_size=(160, 160), 
    dropout_rate=0.4, 
    optimizer=tf.keras.optimizers.legacy.Adam()):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    input_shape=target_size + (3,)
    seed_init = seed
    base_model = model_class(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # freeze base model
    base_model.trainable = False
    # set base model to inference mode (not training)
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # add head comparable to original MobileNetV2
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # add additional layers in case of dropout is not None
    if dropout_rate is not None:
        # Add a fully-connected layer
        seed_init += 1
        initializer = tf.keras.initializers.GlorotUniform(seed=seed_init)
        x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    seed_init += 1
    initializer = tf.keras.initializers.GlorotUniform(seed=seed_init)
    outputs = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train(
    model,
    ds_train, 
    ds_val,
    seed, 
    target_size, 
    batch_size=64,
    epochs=20,
    callbacks=None,
    ):
    tf.keras.utils.set_random_seed(seed)
    ds_train_prepared = prepare_dataset(
        ds_train, seed, target_size=target_size, batch_size=batch_size, augment=True, shuffle=True)
    ds_val_prepared = prepare_dataset(
        ds_val, seed, target_size=target_size, batch_size=batch_size, augment=False, shuffle=False)

    history = model.fit(
        ds_train_prepared,
        epochs=epochs,
        # batch_size=batch_size,
        validation_data=ds_val_prepared,
        # workers=workers,
        callbacks=callbacks,
    )
    y_true = np.concatenate([y for x, y in ds_val_prepared], axis=0)
    y_pred = np.argmax(model.predict(ds_val_prepared), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    return model, history, cm



def main(argv):
    if tf.config.list_physical_devices('GPU'):
        from numba import cuda
        logging.info("GPU is available")
        device = cuda.get_current_device()
        device.reset()
    else:
        logging.info("GPU not available")


    name_model = 'model_rock_paper_scissors.h5'
    path_model_dir = os.path.join(workspace_dir, 'models')
    path_model_weights_dir = os.path.join(path_model_dir, 'weights')
    path_model = os.path.join(path_model_dir, name_model)

    seed = 1234
    os.environ['PYTHONHASHSEED'] = '0'
    tf.keras.utils.set_random_seed(seed)

    logging.info('loading dataset')
    # load dataset
    dataset = Dataset(seed=seed, dataset_name='rock_paper_scissors')
    dataset.load(validation_proportion=0.15)

    ds_train = dataset.ds_train
    ds_val = dataset.ds_val
    ds_test = dataset.ds_test

    model_class=tf.keras.applications.MobileNetV2
    target_size = (160, 160)
    batch_size = 64

    ds_test_prepared = prepare_dataset(
        ds_test, seed, target_size=target_size, batch_size=batch_size, augment=False, shuffle=False)

    run_training = flags.FLAGS.run_training
    if run_training:
        shutil.rmtree(path_model_dir, ignore_errors=True)
        epochs = 20


        # Create a callback that saves the model's weights
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(path_model_weights_dir, f'model_{datetime.now():%Y%m%d_%H%M%S}'+'-{epoch:02d}-{val_accuracy:.4f}.h5'), 
            monitor='val_accuracy',
            save_weights_only=True,
            verbose=1, 
            save_best_only=True, 
            mode='auto')

        model = create_model(
            model_class=model_class,
            seed=seed,
            target_size=target_size, 
            dropout_rate=0.4, 
            optimizer=tf.keras.optimizers.legacy.Adam(1e-3))

        model, history_val, cm_val = train(
            model=model,
            ds_train=ds_train, 
            ds_val=ds_val, 
            seed=seed, 
            target_size=target_size, 
            batch_size=batch_size, 
            epochs=epochs,
            callbacks=[checkpoint],)

        # load best model
        model.save(path_model)
        loss, accuracy = model.evaluate(ds_test_prepared)
        y_true = np.concatenate([y for x, y in ds_test_prepared], axis=0)
        y_pred = np.argmax(model.predict(ds_test_prepared), axis=1)
        cm_test = confusion_matrix(y_true, y_pred)

        logging.info(f'Performance on test split, after final epoch: {loss}')
        logging.info(f'test loss: {loss}')
        logging.info(f'test accuracy: {accuracy}')
        logging.info(f'test confusion matrix:\n{cm_test}')

        logging.info('Loading weights for best epoch')

        path_model_weights = os.path.join(path_model_weights_dir, sorted(os.listdir(path_model_weights_dir))[-1])
        logging.info(f'Model weights: {path_model_weights}')
        model.load_weights(path_model_weights)
        logging.info('Overwriting model weights with those from best epoch')
        model.save(path_model)

    model = tf.keras.models.load_model(path_model)
    loss, accuracy = model.evaluate(ds_test_prepared)
    y_true = np.concatenate([y for x, y in ds_test_prepared], axis=0)
    y_pred = np.argmax(model.predict(ds_test_prepared), axis=1)
    cm_test = confusion_matrix(y_true, y_pred)

    logging.info(f'Performance on test split, best epoch: {loss}')
    logging.info(f'test loss: {loss}')
    logging.info(f'test accuracy: {accuracy}')
    logging.info(f'test confusion matrix:\n{cm_test}')

    if flags.FLAGS.saved_model:
        tf.saved_model.save(model, os.path.join(path_model_dir, 'rock_paper_scissors-saved_model'))
    

flags.DEFINE_boolean('run_training', default=True, help='whether to run training')
flags.DEFINE_boolean('saved_model', default=True, help='whether to save model as saved model')

if __name__ == '__main__':
    # set logging level
    logging.basicConfig(level=logging.INFO)
    logging.info('running script')
    app.run(main)


