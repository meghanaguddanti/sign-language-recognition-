import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from PIL import Image
from predict import preprocess  # type: ignore[import-not-found]

def make_img(w=100, h=100):
    return Image.fromarray((np.random.rand(h, w, 3)*255).astype("uint8"))

def test_shape():
    assert preprocess(make_img()).shape == (1, 64, 64, 3)
    print("PASS: shape")

def test_normalised():
    r = preprocess(make_img())
    assert r.min() >= 0.0 and r.max() <= 1.0
    print("PASS: normalised")

def test_numpy_input():
    arr = (np.random.rand(80, 80, 3)*255).astype("uint8")
    assert preprocess(arr).shape == (1, 64, 64, 3)
    print("PASS: numpy input")

if __name__ == "__main__":
    test_shape(); test_normalised(); test_numpy_input()
    print("\nAll tests passed ✓")