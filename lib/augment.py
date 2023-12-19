import os
import numpy as np
import tensorflow as tf
import random

class Processing():
    def __init__(
        self,
        random_flip_left_right=True,
        random_brightness=0.1,
        random_contrast=0.1,
        random_rotation=45,
        random_saturation=0.1,
        random_hue=0.1,
        random_crop=0.1,
        random_translate=0.1,
        random_zoom=0.1,
        ):
        self.random_flip_left_right = random_flip_left_right
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_rotation = random_rotation
        self.random_saturation = random_saturation
        self.random_hue = random_hue
        self.random_crop = random_crop
        self.random_translate = random_translate
        self.random_zoom = random_zoom

        self.rotation_layer = None
        if self.random_rotation is not None:
            rotation_factor = self.random_rotation / 360
            self.rotation_layer = tf.keras.layers.RandomRotation(
                factor=(-rotation_factor, rotation_factor), 
                fill_mode='nearest')
        self.translation_layer = None
        if self.random_translate is not None:
            self.translation_layer = tf.keras.layers.RandomTranslation(
                height_factor=[-random_translate, random_translate],
                width_factor=[-random_translate, random_translate],
                fill_mode='nearest')
        self.zoom_layer = None
        if self.random_zoom is not None:
            self.zoom_layer = tf.keras.layers.RandomZoom(
                height_factor=[-random_zoom, random_zoom],
                width_factor=[-random_zoom, random_zoom],
                fill_mode='nearest')
            
    def normalize_image_uint8_to_0_1(self, image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 255.0)
        return image, label
    
    def normalize_image_0_1_to_m1_p1(self, image, label):
        image = tf.cast(image, tf.float32)
        image = (image * 2) -1
        return image, label

    def resize_image(self, image, label, target_size=(224, 224)):
        image = tf.image.resize(image, target_size)
        return image, label

    def augment_image(
            self,
            image, 
            label
            ):
        # Randomly apply augmentations
        
        if self.random_flip_left_right:
            image = tf.image.random_flip_left_right(image)
        if self.random_hue is not None:
            image = tf.image.random_hue(image, max_delta=self.random_hue)
        if self.random_saturation is not None:
            image = tf.image.random_saturation(image, lower=1-self.random_saturation, upper=1+self.random_saturation)
        if self.random_brightness is not None:
            image = tf.image.random_brightness(image, max_delta=0.1)
        if self.random_contrast is not None:
            image = tf.image.random_contrast(image, lower=1-self.random_contrast, upper=1+self.random_contrast)
        if self.random_rotation is not None and self.random_rotation != 0:
            image = self.rotation_layer(image)
        if self.random_crop is not None:
            # retrieve the image size and convert the value to int
            image_size = list(image.shape)[0]
            crop_size = np.round(image_size * (1.0-self.random_crop), 0).astype(np.int32)
            image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
        if self.random_translate is not None:
            image = self.translation_layer(image)
        if self.random_zoom is not None:
            image = self.zoom_layer(image)
        return image, label

    def revert_normalize_0_1_to_uint8(self, image, label):
        # convert back to original value range
        image = image * 255.0
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, 'uint8')
        return image, label


    def preprocess(self, ds, target_size=(224, 224), augment=True, normalize_m1_p1=False):
        # normalize for augmentation [0,1]
        ds_preproc = ds.map(
            self.normalize_image_uint8_to_0_1, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if augment:
            ds_preproc = ds_preproc.map(
                self.augment_image, 
                num_parallel_calls=tf.data.AUTOTUNE
            )
        ds_preproc = ds_preproc.map(
            lambda image, label: self.resize_image(image, label, target_size), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if normalize_m1_p1:
            # normalize for model input [-1,1]
            ds_preproc = ds_preproc.map(
                self.normalize_image_0_1_to_m1_p1, 
                num_parallel_calls=tf.data.AUTOTUNE
            )
        return ds_preproc
    
    def prepare_for_training(
            self, 
            ds, 
            seed=42,
            target_size=(224, 224), 
            batch_size=64, 
            augment=True, 
            shuffle=True,
            cache=True,
            ):
        ds_prepared = ds
        if cache:
            ds_prepared = ds_prepared.cache()
        if shuffle:
            ds_prepared = ds_prepared.shuffle(len(ds), seed=seed)
        ds_prepared = self.preprocess(ds_prepared, target_size=target_size, augment=augment)
        ds_prepared = ds_prepared.batch(batch_size)
        ds_prepared = ds_prepared.prefetch(tf.data.AUTOTUNE)
        return ds_prepared