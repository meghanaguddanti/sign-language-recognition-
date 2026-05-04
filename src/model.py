import os, sys
sys.path.insert(0, os.path.dirname(__file__))

try:
    from tensorflow.keras import layers, models, applications, optimizers
except ImportError:
    from keras import layers, models, applications, optimizers
from config import IMG_SIZE, LEARNING_RATE

def build_model(num_classes: int, fine_tune: bool = False):
    input_shape = (*IMG_SIZE, 3)
    base = applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    if fine_tune:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
    else:
        base.trainable = False

    inputs  = layers.Input(shape=input_shape)
    x       = base(inputs, training=fine_tune)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="SignLanguageCNN")
    lr    = LEARNING_RATE / 10 if fine_tune else LEARNING_RATE
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model