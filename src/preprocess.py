import os, sys
from importlib import import_module
sys.path.insert(0, os.path.dirname(__file__))

try:
    ImageDataGenerator = import_module("tensorflow.keras.preprocessing.image").ImageDataGenerator
except Exception:
    # Fallback to standalone Keras if tensorflow.keras is unavailable
    ImageDataGenerator = import_module("keras.preprocessing.image").ImageDataGenerator
from config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED

def get_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        shear_range=0.05,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,
        fill_mode="nearest",
        validation_split=VAL_SPLIT
    )
    eval_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", shuffle=True, seed=SEED
    )
    val_gen = train_aug.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", shuffle=False, seed=SEED
    )
    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False
    )
    class_names = list(train_gen.class_indices.keys())
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    return train_gen, val_gen, test_gen, class_names