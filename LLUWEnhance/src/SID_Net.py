from __future__ import print_function

import os
import time
import random

from PIL import Image
from numpy.lib.function_base import gradient
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.ops.math_ops import reduce_mean

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import *
import matplotlib.pyplot as pltpipm




def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, channel=64, kernel_size=3):

    #img_data = image.img_to_array(input_im)
    #img_data = np.expand_dims(input_im, axis=0)
    

    # with tf.compat.v1.variable_scope('DecomNet', reuse=tf.compat.v1.AUTO_REUSE):
    #     conv = tf.compat.v1.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")
    #     for idx in range(5):
    #         conv = tf.compat.v1.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_%d' % idx)
    #     conv = tf.compat.v1.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer')


    # input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    # input_im = concat([input_max, input_im])
    with tf.compat.v1.variable_scope('DecomNet', reuse=tf.compat.v1.AUTO_REUSE):
        #conv = tf.compat.v1.layers.conv2d(input_im, 5, kernel_size, padding='same', activation=tf.nn.sigmoid, name='Feature')
        #conv2 = tf.compat.v1.layers.conv2d(input_im, 5, kernel_size, padding='same', activation=tf.nn.tanh, name='FeatureNoise')
        # rNetModel = ResNet50(False, weights='imagenet')
        # img_data = preprocess_input(input_im)
        # feature = rNetModel.predict(img_data, steps=1)

        feature = tf.compat.v1.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")

        conv = tf.compat.v1.layers.conv2d(feature, channel, kernel_size, padding='same', activation=tf.nn.sigmoid, name='Feature')
        conv = tf.compat.v1.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer1')

        conv2 = tf.compat.v1.layers.conv2d(feature, channel, kernel_size, padding='same', activation=tf.nn.tanh, name='FeatureNoise')
        conv2 = tf.compat.v1.layers.conv2d(conv, 1, kernel_size, padding='same', activation=None, name='recon_layer2')
    
    R = conv[:,:,:,0:3]
    L = conv[:,:,:,3:4]
    N = conv2[:,:,:,0:1]
    # k = 3
    # size=input_im.shape[1]
    # input_shape = input_im.shape
    # #tf.reshape(input_im.shape, [input_im.shape[0], input_im.shape[1]*size, input_im.shape[2], input_im.shape[3]])
    # input_shape = [input_im.shape[0], input_im.shape[1]*size, input_im.shape[2], input_im.shape[3]]
    # image_belt = np.zeros(input_shape)
    # #(input_im.shape[0], input_im.shape[1], input_im.shape[2], input_im.shape[3])
    # for i in range(k):
    #     feature_image = conv2[:, :, :, i]
    #     #feature_image-= feature_image.mean()
    #     #feature_image/= feature_image.std ()
    #     #feature_image*=  64
    #     #feature_image+= 128
    #     feature_image= np.clip(input_im, 0, 255).astype('uint8')        
    #     image_belt[:, i * size : (i + 1) * size] = feature_image  

    # scale = 20. / k
    # cv2.imwrite("op.png", image_belt)
    # plt.figure( figsize=(scale * k, scale) )
    # plt.grid  ( False )
    # plt.imshow( image_belt, aspect='auto')
       



    return R, L, N

def RelightNet(input_L, input_R, channel=64, kernel_size=3):
    # input_im = concat([input_R, input_L])
    # with tf.compat.v1.variable_scope('RelightNet'):
    #     conv0 = tf.compat.v1.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
    #     conv1 = tf.compat.v1.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    #     conv2 = tf.compat.v1.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    #     conv3 = tf.compat.v1.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
    #     up1 = tf.compat.v1.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    #     deconv1 = tf.compat.v1.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
    #     up2 = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    #     deconv2= tf.compat.v1.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
    #     up3 = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    #     deconv3 = tf.compat.v1.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
    #     deconv1_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    #     deconv2_resize = tf.compat.v1.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    #     feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
    #     feature_fusion = tf.compat.v1.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
    #     output = tf.compat.v1.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)
    #L_new = 1 - (1 - L_old)n
    #n = log(1 - 0.8)/ log(1 - T)
    #T = clustering pixels of the low-light illumination map into two clusters, bright pixels and dark pixels, by KMeans. 
    #Then taking the minimum illumination value of bright pixels as threshold T,  is calculated
    #Z = tf.cast(input_L, np.float32)
    pca = PCA(n_components=1, random_state=22)
    pca.fit(input_L)
    x = pca.transform(input_L)
    #input_L.numpy()
    #np.float32(input_L)
    #https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(x)

    #c10riteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #ret,label,center = cv2.kmeans(cv2.UMat(Z), 2, None, criteria, 20, flags)

   


    #gives cluster with bright points
    bright = 0
    #Z[label.ravel()==0]
    T = bright.min()
    n = tf.math.log( 1 - 0.8)/ tf.math.log(1 - T)

    #need to convert n to tensor
    L_new = 1 - tf.pow((1 - input_L), n) 
    output = concat([input_R, L_new])

    return output

# def input_fn():
#   return tf.compat.v1.train.limit_epochs(
#       tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)


class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        
        self.input_low = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_he = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        [R_low, I_low, N_low] = DecomNet(self.input_low)
        [R_he, I_he, N_he] = DecomNet(self.input_he)
        
        # I_delta = RelightNet(I_low, R_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_he, I_he, I_he])
        # I_delta_3 = concat([I_delta, I_delta, I_delta])

        self.output_R_low = R_low
        self.output_I_low = I_low_3
        # self.output_I_delta = I_delta_3
        # self.output_S = R_low * I_delta_3


        #Reflectance consistency loss
        #Lrc = ||Rlow - Rhe||1
        Lrc = tf.norm( R_low - R_he, ord=1)
            
        #retinex reconstruction loss
        #Lrec = ||R*I+N-S||1
        Lrec_low = tf.norm( R_low * I_low + N_low - self.input_low, ord=1 )
        Lrec_he = tf.norm( R_he * I_he + N_he - self.input_he, ord=1)

        print(tf.shape(R_low))
        print(tf.shape(self.input_low))
        print(tf.shape(R_he))
        print(tf.shape(self.input_he))
        print(R_low)
        print(self.input_low)
        print(R_he)
        print(self.input_he)


        #Illumination smoothness and consistency loss.
        #LI = ||∇I * exp(-α∇R)||1 + ||∇I * exp(-α ∑i=low,he ∇I_i)||1
        #exp exponential distribution 	f (x) = λe-λx , x≥0
        LI_low = self.smooth_sid(I_low, I_he, I_low, R_low)
        LI_he = self.smooth_sid(I_low, I_he, I_he, R_he)


        print(tf.shape(R_low))
        print(tf.shape(self.input_low))
        print(tf.shape(R_he))
        print(tf.shape(self.input_he))
        print(R_low)
        print(self.input_low)
        print(R_he)
        print(self.input_he)

        #Reflectance contrast and color loss
        #LR = 1/3 ∑ch=R,G,B ||∇Rch - β∇Sch||F + ||RH -SH||2 
        #β is the gradient amplification factor which is set as 10 in experiments
        #∇Sch s a variant of the gradient of the input image ∇S
        #∇Sch = { =0 if |∇S| <  ε else = ∇S }
        #RH and SH are the hue channels of the reflectance map and the source image after converting them from the RGB to the HSV color space.
        #get gradient of S ∇S 
        LR_low = self.reflectanceContrast(R_low, self.input_low, R_low )
        LR_he = self.reflectanceContrast(R_he, self.input_he, R_low )


        print(tf.shape(R_low))
        print(tf.shape(self.input_low))
        print(tf.shape(R_he))
        print(tf.shape(self.input_he))
        print(R_low)
        print(self.input_low)
        print(R_he)
        print(self.input_he)




        #Noise estimation loss
        #LN = ||S * N||F
        LN_low = tf.norm(self.input_low * N_low)
        # tf.reduce_mean(tf.abs(self.input_low * N_low))
        LN_he =  tf.norm(self.input_he * N_he)
        # tf.reduce_mean(tf.abs(self.input_he * N_he))


        #total loss
        #L = λrc * Lrc + ∑i=low,he (Lrec i + λI * LI i + λR * LR i + λN * LN i)
        #λrc  λI  λR  λN coeff 0.01, 0.1, 0.001, 0.01
        lambdaRC = 0.01
        lambdaI = 0.1
        lambdaR = 0.001
        lambdaN = 0.01

        TotalLoss = (
                        lambdaRC * Lrc + 
                        Lrec_low + Lrec_he +
                        LI_low * lambdaI +  LI_he * lambdaI +
                        LR_low * lambdaR +  LR_he * lambdaR + 
                        LN_low * lambdaN +  LN_he * lambdaN
                    )
        

        print(TotalLoss)

        # loss
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_he * I_high_3 - self.input_he))
        self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_he * I_low_3 - self.input_low))
        self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - self.input_he))
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_he))
        # self.relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_3 - self.input_high))

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_he, R_he)
        # self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_Decom = TotalLoss
        # self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss
        # self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta

        self.lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.compat.v1.trainable_variables() if 'DecomNet' in var.name]
        # self.var_Relight = [var for var in tf.compat.v1.trainable_variables() if 'RelightNet' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
        # self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver_Decom = tf.compat.v1.train.Saver(var_list = self.var_Decom)
        # self.saver_Relight = tf.compat.v1.train.Saver(var_list = self.var_Relight)

        print("[*] Initialize model successfully...")
    #     self.sess = sess
    #     self.DecomNet_layer_num = 5

    #     # build the model
    #     self.input_low = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_low')
    #     self.input_he = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_he')

    #     [R_low, I_low, N_low] = DecomNet(self.input_low)
    #     [R_he, I_he, N_he] = DecomNet(self.input_he)
        
    #    # I_delta = RelightNet(I_low, R_low)

    #     I_low_3 = concat([I_low, I_low, I_low])
    #     N_low_3 = concat([N_low, N_low, N_low])
    #     # I_high_3 = concat([I_he, I_he, I_he])
    #    # I_delta_3 = concat([I_delta, I_delta, I_delta])

    #     self.output_R_low = R_low
    #     self.output_I_low = I_low_3
    #     self.output_N_low = N_low_3
    #   #  self.output_I_delta = I_delta_3
    #    # self.output_S = R_low * I_delta_3

  

    #     #Reflectance consistency loss
    #     #Lrc = ||Rlow - Rhe||1
    #     Lrc = tf.norm( R_low - R_he, ord=1)
    #     # tf.reduce_mean( tf.abs(R_low - R_he) )

    #     #retinex reconstruction loss
    #     #Lrec = ||R*I+N-S||1
    #     Lrec_low = tf.norm( R_low * I_low + N_low - self.input_low, ord=1 )
    #     # tf.reduce_mean( tf.abs(R_low * I_low + N_low - self.input_low) )
    #     Lrec_he = tf.norm( R_he * I_he + N_he - self.input_he, ord=1)
    #     # tf.reduce_mean( tf.abs(R_he * I_he + N_he - self.input_he) )

    #     #Illumination smoothness and consistency loss.
    #     #LI = ||∇I * exp(-α∇R)||1 + ||∇I * exp(-α ∑i=low,he ∇I_i)||1
    #     #exp exponential distribution 	f (x) = λe-λx , x≥0
    #     LI_low = self.smooth_sid(I_low, I_he, I_low, R_low)
    #     LI_he = self.smooth_sid(I_low, I_he, I_he, R_he)

    #     #Reflectance contrast and color loss
    #     #LR = 1/3 ∑ch=R,G,B ||∇Rch - β∇Sch||F + ||RH -SH||2 
    #     #β is the gradient amplification factor which is set as 10 in experiments
    #     #∇Sch s a variant of the gradient of the input image ∇S
    #     #∇Sch = { =0 if |∇S| <  ε else = ∇S }
    #     #RH and SH are the hue channels of the reflectance map and the source image after converting them from the RGB to the HSV color space.
    #     #get gradient of S ∇S 
    #     LR_low = self.reflectanceContrast(R_low, self.input_low, R_low )
    #     LR_he = self.reflectanceContrast(R_he, self.input_he, R_low )

    #     #Noise estimation loss
    #     #LN = ||S * N||F
    #     LN_low = tf.norm(self.input_low * N_low)
    #     # tf.reduce_mean(tf.abs(self.input_low * N_low))
    #     LN_he =  tf.norm(self.input_he * N_he)
    #     # tf.reduce_mean(tf.abs(self.input_he * N_he))


    #     #total loss
    #     #L = λrc * Lrc + ∑i=low,he (Lrec i + λI * LI i + λR * LR i + λN * LN i)
    #     #λrc  λI  λR  λN coeff 0.01, 0.1, 0.001, 0.01
    #     lambdaRC = 0.01
    #     lambdaI = 0.1
    #     lambdaR = 0.001
    #     lambdaN = 0.01

    #     TotalLoss = (
    #                     lambdaRC * Lrc + 
    #                     Lrec_low + Lrec_he +
    #                     LI_low * lambdaI +  LI_he * lambdaI +
    #                     LR_low * lambdaR +  LR_he * lambdaR + 
    #                     LN_low * lambdaN +  LN_he * lambdaN
    #                 )
        


    #     # loss
    #     #self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  self.input_low))
    #     #self.recon_loss_high = tf.reduce_mean(tf.abs(R_he * I_high_3 - self.input_he))
    #     #self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_he * I_low_3 - self.input_low))
    #     #self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - self.input_he))

    #     #self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_he))
    #     #self.relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_3 - self.input_he))

    #     #self.Ismooth_loss_low = self.smooth(I_low, R_low)
    #     #self.Ismooth_loss_high = self.smooth(I_he, R_he)

    #    # self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

    #     #self.loss_Decom = self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss
    #     self.loss_Decom = TotalLoss
    #     #self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta

    #     self.lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
    #     optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')

    #     self.var_Decom = [var for var in tf.compat.v1.trainable_variables() if 'DecomNet' in var.name]
    #     #self.var_Relight = [var for var in tf.compat.v1.trainable_variables() if 'RelightNet' in var.name]

    #     self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
    #     #self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)

    #     self.sess.run(tf.compat.v1.global_variables_initializer())

    #     self.saver_Decom = tf.compat.v1.train.Saver(var_list = self.var_Decom)
    #     #self.saver_Relight = tf.compat.v1.train.Saver(var_list = self.var_Relight)

    #     print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])

        # print(tf.shape(self.smooth_kernel_x))

        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
            # kernel = [-1, 1]
        elif direction == "y":
            kernel = self.smooth_kernel_y
            # kernel = [1, -1]
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    # def gradient3(self, input_tensor, direction):
    #     self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0, 0 ,0],[0, 0, 0 ,0], [-1, 1, 1, 1]], tf.float32), [2, 2, 1, 3])
    #     self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

    #     if direction == "x":
    #         kernel = self.smooth_kernel_x
    #     elif direction == "y":
    #         kernel = self.smooth_kernel_y
    #     return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.compat.v1.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    #Illumination smoothness and consistency loss.
    #LI = ||∇I * exp(-α∇R)||1 + ||∇I * exp(-α ∑i=low,he ∇I_i)||1
    def smooth_sid(self, Ilow, Ihe, I, R ):
        alpha = 10
        gradI_x = self.gradient(I, "x")
        gradI_y = self.gradient(I, "y")

        avggradR_x = self.ave_gradient(tf.image.rgb_to_grayscale(R), "x")
        avggradR_y = self.ave_gradient(tf.image.rgb_to_grayscale(R), "y")

        avggradIlow_x = self.ave_gradient(Ilow, "x")
        avggradIlow_y = self.ave_gradient(Ilow, "y")

        avggradIhe_x = self.ave_gradient(Ihe, "x")
        avggradIhe_y = self.ave_gradient(Ihe, "y")

        #||∇I * exp(-α∇R)||1 
        P1 = tf.norm( gradI_x * tf.exp( -alpha * avggradR_x) + gradI_y * tf.exp( -alpha * avggradR_y) , ord=1)
        #||∇I * exp(-α ∑i=low,he ∇I_i)||1
        P2 = tf.norm( gradI_x * tf.exp( -alpha * ( avggradIlow_x + avggradIhe_x ) ) + gradI_y * tf.exp( -alpha * ( avggradIlow_y + avggradIhe_y ) ) , ord=1)
        return P1 + P2

    @tf.function
    def getVariantofS(self, gradS_X):
        # pred = tf.less(gradS_X, tf.constant([tf.keras.backend.epsilon()]))
        
        

        # val_if_true = tf.constant(0, dtype=tf.float32)
        # val_if_false = gradS_X
        # result = tf.where(pred, val_if_true, val_if_false)

        # with tf.compat.v1.Session() as sess:
        #     sess.run(result, feed_dict={pred: True})   # ==> 28.0
        #     sess.run(result, feed_dict={pred: False})  # ==> 12.0


        # tf.less(gradS_X, tf.constant([tf.keras.backend.epsilon()]))
         if( gradS_X < tf.keras.backend.epsilon()):
             return tf.constant(0, dtype=tf.float32)
         else:
             return gradS_X
    
    # def check_image(self, image):
    #     assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    #     with tf.control_dependencies([assertion]):
    #         image = tf.identity(image)

    #     if image.get_shape().ndims not in (3, 4):
    #         raise ValueError("image must be either 3 or 4 dimensions")

    #     # make the last dimension 3 so that you can unstack the colors
    #     shape = list(image.get_shape())
    #     shape[-1] = 3
    #     image.set_shape(shape)
    #     return image


    #Reflectance contrast and color loss
    #LR = 1/3 ∑ch=R,G,B ||∇Rch - β∇Sch||F + ||RH -SH||2 
    def reflectanceContrast(self, R, S, R_low ):
        beta = 10

        # tf.shape(R)
        
        # tf.shape(S)
        # https://github.com/ry/tensorflow-vgg16/blob/master/vgg16.py#L5
        # rgb_scaled = R * 255.0
        
       
       
        channels = tf.unstack (R, axis=-1)
        red, green, blue = channels[0], channels[1], channels[2]  


        red, green, blue = tf.split(R,3, axis=-1)
       

        R_r = red
        # R[:,:,:,0:1]
        # R[:,:,:,0:1]
        R_g = green
        # R[:,:,:,1:2]
        # R[:,:,:,1:2]
        R_b = blue
        # R[:,:,:,2:3]    
        #  R[:,:,:,2:3]    

        # tf.split(R, [,,-1], axis=1)
        # R,R,R
        # = tf.split(R, 3, axis=1)
        # cv2.split(R)
        
        # print("Test")
        # print(tf.shape(R_r))
        # print(tf.shape(R_g))
        # print(tf.shape(R_b))

        # print("Test")
        # print(tf.shape(R[:,:,:,0:1]))

        # blue, green, red = np.dsplit(R,tf.shape(R))
        
        # print(tf.shape(red))
        # print(tf.shape(green))
        # print(tf.shape(blue))
        # print(red)
        # print(green)
        # print(blue)
        # print("Test")
        # print(R_b)
        # print(R_g)
        # print(R_r)

        red, green, blue = tf.split(S,3, axis=-1)

        S_r = red
        # S[:,:,:,0:1]
        # S[:,:,:,0:1]
        S_g = green
        # S[:,:,:,1:2]
        # S[:,:,:,1:2]
        S_b = blue
        # S[:,:,:,2:3]
        # S[:,:,:,2:3]

        # = tf.split(S, [], axis=1)
        # S,S,S
        # = tf.split(S, 3, axis=1)
        # cv2.split(S)

        # print(tf.shape(red))
        # print(tf.shape(green))
        # print(tf.shape(blue))
        # print(red)
        # print(green)
        # print(blue)

        # print("Test2")
        # print(S_b)
        # print(S_g)
        # print(S_r)

        gradR_r_x = self.gradient(R_r,"x")
        gradR_r_y = self.gradient(R_r,"y")
        gradR_g_x = self.gradient(R_g,"x")
        gradR_g_y = self.gradient(R_g,"y")
        gradR_b_x = self.gradient(R_b,"x")
        gradR_b_y = self.gradient(R_b,"y")

        varS_r_x = self.getVariantofS(self.gradient(S_r,"x"))
        varS_r_y = self.getVariantofS(self.gradient(S_r,"y"))        
        varS_g_x = self.getVariantofS(self.gradient(S_g,"x"))
        varS_g_y = self.getVariantofS(self.gradient(S_g,"y"))
        varS_b_x = self.getVariantofS(self.gradient(S_b,"x"))
        varS_b_y = self.getVariantofS(self.gradient(S_b,"y"))

        P1 = 1/3 * tf.norm( 
                gradR_r_x - ( beta * varS_r_x) + gradR_r_y - (beta * varS_r_y) + 
                gradR_g_x - ( beta * varS_g_x) + gradR_g_y - (beta * varS_g_y) + 
                gradR_b_x - ( beta * varS_b_x) + gradR_b_y - (beta * varS_b_y) )
        
        RH = tf.image.rgb_to_hsv(R_low)
        # cv2.cvtColor(R_low, cv2.COLOR_BGR2HSV)
        SH = tf.image.rgb_to_hsv(self.input_low)
        # cv2.cvtColor(self.input_low, cv2.COLOR_BGR2HSV)
        P2 = tf.norm(tf.abs(RH - SH) , ord=2)

        return P1 + P2

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        # for idx in range(len(eval_low_data)):
        input_low_eval = np.expand_dims(eval_low_data, axis=0)

        if train_phase == "Decom":
            result_1, result_2, result_3 = self.sess.run([self.output_R_low, self.output_I_low, self.output_N_low], feed_dict={self.input_low: input_low_eval})
        # if train_phase == "Relight":
        #     result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta], feed_dict={self.input_low: input_low_eval})

        save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2, result_3)

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        # assert len(train_low_data) == len(train_high_data)
        numBatch = 1 // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            saver = self.saver_Decom
        #elif train_phase == "Relight":
            #train_op = self.train_op_Relight
            #train_loss = self.loss_Relight
            #saver = self.saver_Relight

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    # image_id = (image_id + 1) % len(train_low_data)
                    # if image_id == 0:
                    #     tmp = list(zip(train_low_data, train_high_data))
                    #     random.shuffle(list(tmp))
                    #     train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([train_op, train_loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_he: batch_input_high, \
                                                                           self.lr: lr[epoch]})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.compat.v1.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './model/Decom')
        load_model_status_Relight, _ = self.load(self.saver_Relight, './model/Relight')
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            [R_low, I_low, I_delta, S] = self.sess.run([self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S], feed_dict = {self.input_low: input_low_test})

            if decom_flag == 1:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "_S."   + suffix), S)

