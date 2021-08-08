from glob import glob
import os
import cv2    

# train_low_data_names = glob('./data/our485/low/*.png') + glob('./data/syn/low/*.png')
train_low_data_names = glob('./data/PerformanceCompare_OD/Orignal/*.jpg') 
train_low_data_names.sort() 

print('[*] Number of files found data: %d' % len(train_low_data_names))

for idx in range(len(train_low_data_names)):    
    #create histogram equalised images
    img = cv2.imread(train_low_data_names[idx])      
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))

    directory, filename = os.path.split(os.path.abspath(train_low_data_names[idx]))   
    directory = os.path.dirname(directory) + "\\HE"

    filename = filename.split(".")[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite( directory + '\\he_' + filename+".png", equ)

print('[*] Completed Histogram Equalisation of source Images')