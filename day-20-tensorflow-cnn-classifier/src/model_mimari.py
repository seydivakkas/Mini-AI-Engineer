"""CNN Model Mimari Modülü.

Keras ile Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense ve Dropout
katmanlarını içeren modüler evrişimli sinir ağı (CNN) modelini inşa eder.
"""

from typing import Tuple
import keras
from keras import layers, models, optimizers


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 4,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """Üretim kalitesinde çok katmanlı 2B Evrişimli Sinir Ağı (CNN) modeli kurar ve derler.

    Mimari Yapısı:
    1. Giriş Katmanı: (H, W, C)
    2. Evrişim Bloğu 1: Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
    3. Evrişim Bloğu 2: Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
    4. Evrişim Bloğu 3: Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
    5. Sınıflandırıcı Başlık: Flatten + Dense(128, ReLU) + Dropout(rate) + Dense(num_classes, Softmax)

    Args:
        input_shape: Girdi tensör şekli (yükseklik, genişlik, kanal).
        num_classes: Sınıflandırılacak kategori sayısı.
        learning_rate: Adam optimizer öğrenme oranı.
        dropout_rate: Aşırı öğrenmeyi önleme için bırakma (dropout) oranı.

    Returns:
        keras.Model: Derlenmiş (compiled) Keras modeli.

    Raises:
        ValueError: Girdi boyutları veya sınıf sayısı geçersizse.
    """
    if len(input_shape) != 3 or any(dim <= 0 for dim in input_shape):
        raise ValueError(f"Geçersiz input_shape: {input_shape}")
    if num_classes < 2:
        raise ValueError(f"Sınıf sayısı en az 2 olmalıdır. Verilen: {num_classes}")

    inputs = keras.Input(shape=input_shape, name="girdi_katmani")

    # --- 1. Evrişim Bloğu ---
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv2d_blok1")(inputs)
    x = layers.BatchNormalization(name="bn_blok1")(x)
    x = layers.Activation("relu", name="relu_blok1")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_blok1")(x)

    # --- 2. Evrişim Bloğu ---
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2d_blok2")(x)
    x = layers.BatchNormalization(name="bn_blok2")(x)
    x = layers.Activation("relu", name="relu_blok2")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_blok2")(x)

    # --- 3. Evrişim Bloğu ---
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d_blok3")(x)
    x = layers.BatchNormalization(name="bn_blok3")(x)
    x = layers.Activation("relu", name="relu_blok3")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_blok3")(x)

    # --- Sınıflandırıcı Başlık (Classifier Head) ---
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_gizli")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="olasilik_cikisi")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="day20_vision_cnn")

    # Modeli derle
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
