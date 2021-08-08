# https://github.com/leomauro/image-entropy

import numpy as np
import cv2

def entropy(band: np.ndarray) -> np.float64:
    """
    Compute the entropy content of an image's band.
    :param band: The band data to analyze.
    :return:     Data entropy in bits.
    """
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    # return -np.log2(hist / hist.sum()).sum()
    hist = hist/hist.sum()
    return -(hist * np.log2(hist)).sum()


def show_entropy(band: np.ndarray) -> None:
    """
    Analyze and display entropy of an image's band.
    :param band_name: The name of the band to analyze.
    :param band:      The band data to analyze.
    """
    bits = entropy(band)
    # per_pixel = bits / band.size
    # print(f"{band_name:3s} entropy = {bits:7.2f} bits, {per_pixel:7.6f} per pixel")
    return bits

def getGE_CE(img_file, a_logger:None) -> None:
    """
    Process the image file.
    :param img_file: The image file.
    """
   
    # rgb = img_file.convert("RGB")
    # r, g, b = [np.asarray(component) for component in rgb.split()]
    r = img_file[:,:,2]
    g = img_file[:,:,1]
    b = img_file[:,:,0]    
    
    rEnt = show_entropy(r)
    gEnt = show_entropy(g)
    bEnt = show_entropy(b)

    sumEnt = rEnt + gEnt + bEnt

    if(a_logger != None):
        a_logger.debug('Color entropy of image is :', sumEnt)

    grey = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

    grayEnt = show_entropy(grey)

    if(a_logger != None):
        a_logger.debug('Gray entropy of image is :', grayEnt)

    return grayEnt, sumEnt