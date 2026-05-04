import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR     = os.path.join(BASE_DIR, "data", "raw", "train")
TEST_DIR      = os.path.join(BASE_DIR, "data", "raw", "test")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "best_model.keras")
LABELS_PATH   = os.path.join(MODEL_DIR, "class_names.json")
CURVES_DIR    = MODEL_DIR

IMG_SIZE      = (64, 64)
BATCH_SIZE    = 32
EPOCHS        = 30
VAL_SPLIT     = 0.2
SEED          = 42
LEARNING_RATE = 1e-3
NUM_CLASSES   = None