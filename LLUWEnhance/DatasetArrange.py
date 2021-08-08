import glob
import shutil
import os

ori_path = 'data/Dataset_Part1/'
des_path = 'data/Dataset_Part1/All/'

def move_file(ori_path, des_path):
    
    dir_list = [x[0] for x in os.walk(ori_path)]

    if(not os.path.exists(des_path)):
        os.makedirs(des_path)

    for p in dir_list[1:]:
        for f in glob.glob(p + "/*.JPG"):           
            name = f.split("/")[-2] + "_" + f.split("/")[-1]

            tempPath = os.path.join(des_path,name.split("\\")[0])
            if(not os.path.exists(tempPath)):
                os.mkdir(tempPath)
            shutil.move(f, os.path.join(des_path, name))

move_file(ori_path, des_path)