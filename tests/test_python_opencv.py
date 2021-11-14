from python_opencv.opencv import grayscale_f, is_grayscale
import tempfile
import os

def test_grayscale():
    dirpath = tempfile.mkdtemp()
    output = dirpath + '/gameboy-gray.jpg'
    grayscale_f('./assets/gameboy.jpg', output)
    assert is_grayscale(output)
