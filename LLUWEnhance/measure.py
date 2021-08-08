#!/usr/bin/env python
"""
# > Script for measuring quantitative performances in terms of
#    - Underwater Image Quality Measure (UIQM)
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
## python libs
import os
import ntpath
import numpy as np
# from scipy import misc
import cv2
import logging
import sys
## local libs
from util.data_utils import getPaths

from util.uiqm_utils import getUIQM
from util.ssim_psnr import getSSIM, getPSNR
from util.entropy import getGE_CE
from util.niqe import getNIQE


NIQE_calc, UIQM_calc, GE_CE_calc, ssim_psnr_calc  = True, True, True, True

## data paths
#Lower the better NIQE, 
#Higher PSNR,SSIM,UIQM
#GE,CE closer to orignal

showImageValues = False

defaultImgPath = 'data/PC_0907/'

a_logger = logging.getLogger()
a_logger.setLevel(logging.DEBUG)

output_file_handler = logging.FileHandler("output.log")
stdout_handler = logging.StreamHandler(sys.stdout)

a_logger.addHandler(output_file_handler)
a_logger.addHandler(stdout_handler)


# methods = ["ORIGNAL", "UNET+ZDCE", "UNET+CLAHE", "UNET+CLAHE+ZDCE", "UNET+ZDCE+CLAHE",
#             "UNET", #"L2UWE", "HE", "CLAHE", "ZDCE",
#             #"JED","LIME", "FUNIEGAN", "FGAN+ZDCE", "FGAN+ZDCE+CLAHE",
#             #"FGAN+CLAHE", "FGAN+CLAHE+ZDCE", "ZDCE+FGAN", "ZDCE+UNET", 
#             "l1l2",
#             "l1", "l1+gdl", "l1+msssim", "l1+ssim", "l2",
#             "msssim"
#              ] #name should be same as folder name which images are located

methods = ["ORIGNAL", "unet+clahe+zdce", "unet+zdce", "unet+clahe", "unet+zdce+clahe", "unet"] #n

# resize = [True, False, False, False, False, 
#             False, #True, True, True, True, 
#             #True, True, False, False, False,
#             #False, False, False, False, 
#             False,
#             False, False, False, False, False,
#             False
#             ]

resize = [True, False, False, False, False, False]

## mesures uqim for all images in a directory
def measure_UIQMs(dir_name, enableResize = False):
    paths = getPaths(dir_name)
    uqims = []
    a_logger.debug("UIQM Begin :")
    for img_path in paths:
        im = cv2.imread(img_path)
        if(enableResize):
            im = cv2.resize(im, (256,256))
        val = getUIQM(im)
        if(showImageValues):
            a_logger.debug(os.path.basename(img_path) + " : {0}".format(val) )
        uqims.append(val)
    a_logger.debug("UIQM End")
    a_logger.debug("")
    return np.array(uqims)

## compares avg ssim and psnr 
def measure_SSIM_PSNRs(GT_dir, Gen_dir, resizeBoth = False):
    """
      Assumes:
        * GT_dir contain ground-truths {filename.ext}
        * Gen_dir contain generated images {filename_gen.png}
        * Images are of same-size
    """
    GT_paths = getPaths(GT_dir)
    GEN_paths = getPaths(Gen_dir)
    
    ssims, psnrs, = [], []
    for img_path1, img_path2 in zip(GT_paths, GEN_paths) :
        name_split = ntpath.basename(img_path1).split('.')
        # gen_path = os.path.join(os.getcwd() + "/"+ Gen_dir, resPrepend + name_split[0] + resultExt +'.png') #+name_split[1])
        # if (gen_path in Gen_paths):
        r_im = cv2.imread(img_path1)#BGR        
        r_im = cv2.resize(r_im, (256,256))#convert Gtruth size to 256
        g_im = cv2.imread(img_path2)#bgr
        if(resizeBoth):
             g_im = cv2.resize(g_im, (256,256))#convert Gtruth size to 256

        assert (r_im.shape==g_im.shape), "The images should be of same-size"
        ssim = getSSIM(r_im, g_im)
        psnr = getPSNR(r_im, g_im)

        # GE, CE = getGE_CE(r_im)
        # GE_Grd, CE_Grd = getGE_CE(g_im)
        
        if(showImageValues):
            a_logger.debug ("SSIM for {0} : {1}".format(name_split[0] , ssim))
            a_logger.debug ("PSNR for {0} : {1}".format(name_split[0] , psnr))

        ssims.append(ssim)
        psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)

## compares avg NIQE
def measure_NIQE(Gen_dir, enableResize = False):
    """
      Assumes:
        * GT_dir contain ground-truths {filename.ext}
        * Gen_dir contain generated images {filename_gen.png}
        * Images are of same-size
    """
    Gen_paths = getPaths(Gen_dir)
    nique_rs = []
    for img_path in Gen_paths:
        name_split = ntpath.basename(img_path)
        # gen_path = os.path.join(Gen_dir, 'he_'+name_split[0]+'.jpg') #+name_split[1])
        # if (gen_path in Gen_paths):

        r_im = cv2.imread(img_path)#BGR

        if(enableResize):
            r_im = cv2.resize(r_im, (256,256))
        # g_im = cv2.imread(gen_path)#bgr
        # assert (r_im.shape==g_im.shape), "The images should be of same-size"
        greyR = cv2.cvtColor(r_im, cv2.COLOR_BGR2GRAY)
        # greyG = cv2.cvtColor(r_im, cv2.COLOR_BGR2GRAY)

        # ref = greyR[:,:,0] # ref
        # dis = greyG[:,:,0] # dis
        # niqe_orignal = getNIQE(greyR) 
        niqe_result = getNIQE(greyR)
        
        # print ("{0} : {1}".format(gen_path, niqe_orignal))
        
        if(showImageValues):
            a_logger.debug ("{0} : {1}".format(name_split, niqe_result))
        
        # niqe_os.append(niqe_orignal)
        nique_rs.append(niqe_result)
    return np.array(nique_rs)

def GrayAndColEntropy(dir, enableResize = False):
    GT_paths = getPaths(dir)
    GE_total, CE_total, = [], []
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path)
        r_im = cv2.imread(img_path)#BGR
        if(enableResize):
            r_im = cv2.resize(r_im, (256,256))

        GE, CE = None, None
        if(showImageValues):
            a_logger.debug("CE,GE of {0}".format(name_split))
            GE, CE = getGE_CE(r_im, a_logger)            
            a_logger.debug("\n")
        else:
            GE, CE = getGE_CE(r_im, None)
            

        GE_total.append(GE)
        CE_total.append(CE)

    return GE_total, CE_total

## compares GrayMeanGradient
def measure_GMG(GT_dir):
    GT_paths = getPaths(GT_dir)
    GMG_appended = [], []
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path).split('.')
        r_im = cv2.imread(img_path)#BGR
        greyR = cv2.cvtColor(r_im, cv2.COLOR_BGR2GRAY)

        greyR = cv2.Laplacian(greyR, cv2.CV_64F)
        GMG_appended.append(greyR)
        print(os.path.basename(name_split) + " : {0}".format(greyR) )
    return np.array(GMG_appended)

################################### compute and compare NIQE
if(NIQE_calc):
    a_logger.debug("NIQE")

    i = 1 
    for mName in methods:
        a_logger.debug("\n{0} {1}".format(i, mName))
        NIQE_res_measures = measure_NIQE(defaultImgPath+mName, resize[i-1])
        a_logger.debug ("NIQE_Res >> Mean: {0} std: {1}".format(np.mean(NIQE_res_measures), np.std(NIQE_res_measures)))
        i += 1

################################### compute UIQMs
if(UIQM_calc):
    a_logger.debug("\nUIQM")

    i = 1 
    for mName in methods:
        a_logger.debug("\n{0} {1}".format(i, mName))
        gen_uqims = measure_UIQMs(defaultImgPath+mName, resize[i-1])
        a_logger.debug ("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))
        i += 1

################################### Compute CE & GE
if(GE_CE_calc):
    a_logger.debug("\nCE & GE")

    i = 1 
    for mName in methods:
        a_logger.debug("\n{0} {1}".format(i, mName))
        GE,CE = GrayAndColEntropy(defaultImgPath+mName, resize[i-1])
        a_logger.debug ("Generated GE  >> Mean: {0} std: {1}".format(np.mean(GE), np.std(GE)))
        a_logger.debug ("Generated CE  >> Mean: {0} std: {1}".format(np.mean(CE), np.std(CE)))
        i += 1
    
################################### Calc SSIM PSNR
if(ssim_psnr_calc):
    a_logger.debug("\nSSIM_PSNR")

    i = 1 
    for mName in methods:
        if(i > 1):
            a_logger.debug("\n{0} {1}".format(i, mName))
            SSIM_measures, PSNR_measures = measure_SSIM_PSNRs(defaultImgPath+methods[0], defaultImgPath+mName, resize[i-1])
            a_logger.debug ("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
            a_logger.debug ("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
        i += 1