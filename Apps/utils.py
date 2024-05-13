from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.utils import Sequence, to_categorical, load_img, img_to_array
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
import imgaug.augmenters as iaa
import tensorflow.keras as keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from keras.models import Model
from keras.utils import Sequence
import tensorflow as tf
from keras.preprocessing import image
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate

import numpy as np
from PIL import UnidentifiedImageError
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


labels_colors = {
    0: [(0, 0, 0)],  # void
    1: [(128, 64, 128), (244, 35, 232), (81, 0, 81)],  # flat
    2: [(70, 130, 180)],  # sky
    3: [(220, 20, 60)],  # human
    4: [(119, 11, 32)],  # vehicle
    5: [(220, 220, 0), (153, 153, 153)],  # object
    6: [(70, 70, 70)],  # construction
    7: [(107, 142, 35)]  # nature
    # 8: [(81, 0, 81)] #ground
}

img_height = 256
img_width = 256
classes = len(labels_colors)
n_classes = len(labels_colors)


def get_file_list(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    file_list.sort()
    return file_list


# Définition de dice_loss


@register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    """
   Fonction de perte de Dice pour la segmentation sémantique.
    Args:
    y_true (tensor): Tensor des étiquettes réelles.
    y_pred (tensor): Tensor des prédictions du modèle.
    smooth (float): Terme de lissage pour éviter la division par zéro.
    Returns:
    float: Perte de Dice.
    """

    # Aplatir les tensors

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])

    union = tf.reduce_sum(
        y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])

    dice = (2. * intersection + smooth) / (union + smooth)

    dice_loss = 1 - tf.reduce_mean(dice, axis=-1)

    return dice_loss


class seg_gen(Sequence):
    def __init__(self, x_set, y_set, batch_size=16, with_data_augmentation=False, num_classes=8):
        self.x, self.y = x_set, y_set
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.with_data_augmentation = with_data_augmentation
        if self.with_data_augmentation:
            self.augmenter = self._get_augmenter()

        if self.with_data_augmentation:
            self.augmentation = iaa.Sequential([
                iaa.Fliplr(0.5),  # Horizontal flips
                iaa.Affine(
                    # Random rotations between -10 and 10 degrees
                    rotate=(-10, 10),
                    # Random shearing between -5 and 5 degrees
                    shear=(-5, 5),
                    # Random scaling between 80% and 120%
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
                ),
                # Gaussian blur with random sigma
                iaa.GaussianBlur(sigma=(0, 1.0))
            ])

    def __len__(self):
        return np.ceil(len(self.x) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        batch_x_paths = self.x[idx *
                               self.batch_size:(idx + 1) * self.batch_size]
        batch_y_paths = self.y[idx *
                               self.batch_size:(idx + 1) * self.batch_size]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                self._process_image_and_mask, batch_x_paths, batch_y_paths))

        # Séparez les images et les masques à partir des résultats
        batch_x, batch_y = zip(*results)

        return np.array(batch_x), np.array(batch_y)

    def _process_image_and_mask(self, x_path, y_path):
        """Charge et traite une image et son masque."""
        img = self._load_image(x_path)
        mask = self._load_mask(y_path)

        if self.with_data_augmentation:
            img, mask = self._augment_image_and_mask(img, mask)

        return img, mask

    def _load_image(self, file_name):
        try:
            img = load_img(file_name, target_size=(256, 256))
            img = img_to_array(img)/255
            return img
        except UnidentifiedImageError:
            print(f"Erreur en chargeant l'image : {file_name}")
            raise

    def _load_mask(self, mask_file):
        mask = load_img(mask_file, color_mode='grayscale',
                        target_size=(256, 256))
        mask = img_to_array(mask).astype(int)

        mapped_mask = self._map_labels_to_classes(mask[:, :, 0])

        one_hot_mask = to_categorical(
            mapped_mask, num_classes=self.num_classes)

        return one_hot_mask

    def _map_labels_to_classes(self, mask):

        id_to_category_id_mapping = {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1, 11: 2,
            12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 3, 18: 3, 19: 3, 20: 3, 21: 4,
            22: 4, 23: 5, 24: 6, 25: 6, 26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7,
            32: 7, 33: 7, 34: 7}

        mapped_mask = np.vectorize(id_to_category_id_mapping.get)(mask)

        return mapped_mask

    def _get_augmenter(self):
        """ Create augmentation sequence"""
        return iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 20)),
            iaa.Multiply((0.8, 1.2))
        ])

    def _augment_image_and_mask(self, image, mask):
        # Assurez-vous que le masque est booléen ou entier
        mask = mask.astype(np.int32)

        # Convertissez le masque en SegmentationMapsOnImage
        segmap = SegmentationMapsOnImage(mask, shape=image.shape)

        # Appliquez l'augmentation à l'image et au masque
        image_aug, segmaps_aug = self.augmenter(
            image=image.astype(np.uint8), segmentation_maps=segmap)

        # Convertissez le résultat en format approprié
        image_aug = image_aug.astype(np.float32) / 255.0
        mask_aug = segmaps_aug.get_arr().astype(np.float32)

        return image_aug, mask_aug


def baseline_model(img_height, img_width, nclasses):
    # Assure que les dimensions de l'entrée correspondent à vos images
    inputs = Input(shape=(img_height, img_width, 3))

    # Encodeur
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    dropout1 = Dropout(rate=0.1, seed=42)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    dropout2 = Dropout(rate=0.1, seed=42)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    dropout3 = Dropout(0.1, seed=42)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)

    # Décodeur
    up5 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
        2, 2), padding='same')(conv4), conv3], axis=3)

    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    dropout4 = Dropout(0.1, seed=42)(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
        2, 2), padding='same')(conv6), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    dropout5 = Dropout(rate=0.1, seed=42)(conv8)
    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(conv8), conv1], axis=3)
    conv10 = Conv2D(64, 3, activation='relu', padding='same')(up9)

    # Couche de sortie
    output = Conv2D(nclasses, (1, 1), activation='softmax')(conv10)

    model = Model(inputs=inputs, outputs=output)

    return model


class CustomUNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    @staticmethod
    def conv_block(inputs, filters, kernel_size=3, num_layers=8):
        conv = inputs
        for _ in range(num_layers):
            conv = Conv2D(filters, kernel_size,
                          activation='relu', padding='same')(conv)
        return conv

    @staticmethod
    def encoder_block(inputs, filters, pool_size=(2, 2), num_layers=2):
        conv = CustomUNet.conv_block(inputs, filters, num_layers=num_layers)
        pool = MaxPooling2D(pool_size)(conv)
        return conv, pool

    @staticmethod
    def decoder_block(inputs, skip_features, filters, kernel_size=3, upsample_size=(2, 2), num_layers=2):
        upsample = UpSampling2D(upsample_size)(inputs)
        skip_conv = Conv2D(filters, 1, activation='relu')(skip_features)
        skip_upsample = UpSampling2D(upsample_size)(skip_conv)
        concat = Concatenate(axis=-1)([upsample, skip_upsample])
        conv = CustomUNet.conv_block(concat, filters, num_layers=num_layers)
        return conv

    def build_model(self):
        inputs = Input(self.input_shape)

        conv1, pool1 = self.encoder_block(inputs, 64)
        dropout0 = Dropout(0.1, seed=42)(conv1)
        conv2, pool2 = self.encoder_block(pool1, 128)
        dropout1 = Dropout(rate=0.2, seed=42)(conv2)
        conv3, _ = self.encoder_block(pool2, 256)  # Discard the pool3 output

        bottleneck = self.conv_block(conv3, 512)

        up4 = self.decoder_block(bottleneck, conv3, 256)
        dropout3 = Dropout(rate=0.2, seed=42)(up4)
        up5 = self.decoder_block(up4, conv2, 128)
        dropout2 = Dropout(rate=0.2, seed=42)(up5)
        up6 = self.decoder_block(up5, conv1, 64)

        upsampled_output = UpSampling2D(size=(16, 16))(up6)
        output = Conv2D(8, 1, activation='softmax')(
            upsampled_output)  # 8 classes for segmentation

        model = Model(inputs=inputs, outputs=output)
        return model

    def get_model(self):
        return self.model

# Example usage
# input_shape = (img_height, img_width, 3)  # Example input shape
# custom_unet = CustomUNet(input_shape)
# model = custom_unet.get_model()
# model.summary()
