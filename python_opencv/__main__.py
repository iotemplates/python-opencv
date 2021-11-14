from python_opencv.opencv import *
import sys

if __name__ == "__main__" and len(sys.argv) == 3:
        ocr_preprocess_f(sys.argv[1], sys.argv[2])
else:
    print("Usage: python_opencv <input-image> <output-image>.")
