import cv2
import os
import numpy as np 

def as_image(item):
    """Converts the specified item to image."""
    if type(item) == str:
        if not os.path.isfile(item):
            return False
        return cv2.imread(item)
    return item

def is_grayscale(item):
    """Returns true if item is a grayscale image."""
    img = as_image(item)
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

def grayscale(item):
    """Converts and image to grayscale. Takes item as input and returns it as grayscale."""
    img = as_image(item)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def grayscale_f(in_img, out_img):
    """Grayscale and save the specified image. Takes in_image as input, grayscales and saves to out_img."""
    cv2.imwrite(out_img, grayscale(in_img))

def threshold(item):
    """Threshold the specified image.Takes item as input and returns it as threshold."""
    return cv2.threshold(as_image(item), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def threshold_f(in_img, out_img):
    """Threshold and save the specified image. Takes in_image as input, thresholds and saves to out_img."""
    cv2.imwrite(out_img, threshold(in_img))

def denoise(item, size=5):
    """Denoise the specified image.Takes item as input, denoises it and returns it."""
    return cv2.medianBlur(item, size) 

def denoise_f(in_img, out_img, size=5):
    """Denoise and save the specified image. Takes in_image as input, denoises and saves to out_img."""
    cv2.imwrite(out_img, denoise(in_img, size))

def dilate(item, size=5, iterations=1):
    """Dilate the specified image.Takes item as input, dilates it and returns it."""
    kernel = np.ones((size,sizw),np.uint8)
    return cv2.dilate(as_image(item), kernel, iterations = 1)

def dilate_f(in_img, out_img, size=5, iterations=1):
    """Dilate and save the specified image. Takes in_image as input, dilates and saves to out_img."""
    cv2.imwrite(out_img, dilate(in_img, size, iterations))

def erode(item, size=5, iterations=1):
    """Erode the specified image.Takes item as input, erodes it and returns it."""
    kernel = np.ones((size,sizw),np.uint8)
    return cv2.erode(as_image(item), kernel, iterations = 1)

def erode_f(in_img, out_img, size=5, iterations=1):
    """Erode and save the specified image. Takes in_image as input, erodes and saves to out_img."""
    cv2.imwrite(out_img, erode(in_img, size, iterations))

def opening(item, size=5):
    """Erode/Dilate the specified image.Takes item as input, erodes-dilates (opening) it and returns it."""
    kernel = np.ones((size,sizw),np.uint8)
    return cv2.morphologyEx(as_image(item), cv2.MORPH_OPEN, kernel)

def opening_f(in_img, out_img, size=5, iterations=1):
    """Erode/Dilate and save the specified image. Takes in_image as input, denoises and saves to out_img."""
    cv2.imwrite(out_img, erode(in_img, size, iterations))

def canny(item):
    """Perform canny edge detect and save the specified image. Takes image as input, performs canny edge detection and returns it."""
    return cv2.Canny(as_image(item), 100, 200)

def canny_f(in_img, out_img):
    """Perform canny edge detection the specified image.Takes item as input, performs cunny edge detection it and returns it."""
    cv2.imwrite(out_img, canny(in_img))

def ocr_preprocess(item):
    """Preprocess item for OCR. Takes item as input and performs a series of preprocessing steps for OCR, returning the preprocessed image."""
    return canny(denoise(threshold(grayscale(item))));

def ocr_preprocess_f(in_img, out_img):
    """Preprocess item for OCR and save thre result. Takes item as input and performs a series of preprocessing steps for OCR. It saves the result to out_img."""
    cv2.imwrite(out_img, ocr_preprocess(in_img))
