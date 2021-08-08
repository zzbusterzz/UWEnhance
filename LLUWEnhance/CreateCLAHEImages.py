from glob import glob
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tf_clahe    

# train_low_data_names = glob('./data/our485/low/*.png') + glob('./data/syn/low/*.png')
train_low_data_names = glob('./data/PerformanceCompare/FUNIEGAN/*.jpg') 
train_low_data_names.sort() 

print('[*] Number of files found data: %d' % len(train_low_data_names))

for idx in range(len(train_low_data_names)): 
    data_lowlight_path = train_low_data_names[idx]
    original_img = Image.open(data_lowlight_path)
    original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])

    img_lowlight = tf.io.decode_image(tf.io.read_file(data_lowlight_path))
    enhance_image = tf_clahe.clahe(img_lowlight)

    enhance_image = Image.fromarray(enhance_image.numpy())
    enhance_image = enhance_image.resize(original_size, Image.ANTIALIAS)    

    directory, filename = os.path.split(os.path.abspath(train_low_data_names[idx]))   
    directory = os.path.dirname(directory) + "\\FGAN+CLAHE"

    if not os.path.exists(directory):
        os.makedirs(directory)

    enhance_image.save(directory + '\\' + filename.replace(".jpg", "_clahe.png"))

print('[*] Completed CLAHE of source Images')