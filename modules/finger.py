"""
Just a finger class
"""
from PIL import Image

class Finger:
    """
    Finger class
    """
    def __init__(self, imgpath):
        self.image = Image.open(imgpath)
