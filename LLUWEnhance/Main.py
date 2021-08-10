from __future__ import print_function
import ntpath
import re

import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
import os
import argparse

import time
import imageio

from glob import glob
from PIL import Image
from tensorflow.python.ops.numpy_ops.np_math_ops import _tf_gcd

import multiprocessing
import tf_clahe

# Local imports
from src.unet_model import UNet
from src.zeroDCE import zeroDECModel
from src.loss import *
from util.utils import *

parser = argparse.ArgumentParser(description='')

# GPU arguments
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="0 to 1, gpu memory usage")

# Train/test agrument
parser.add_argument('--phase', choices=['train', 'test'], default='train', help='test or test')

# Unet argument
parser.add_argument('--train_batch_size_ie',        dest='train_batch_size',            type=int,   default=16,                         help='Batch size of image enchancement train')
parser.add_argument('--test_batch_size_ie',         dest='test_batch_size',             type=int,   default=1,                         help='Batch size of image enchancement test')
parser.add_argument('--num_epochs_ie',              dest='num_epochs_ie',               type=int,   default=200,                         help="Number of epoch for Image Enhancement")
parser.add_argument('--lr',                         dest='lr',                          type=float, default=0.0001,                     help="Learning Rate" )

parser.add_argument('--trainA_path',                dest='trainA_path',                 type=str,   default='./data/Water/',            help="Water images path")
parser.add_argument('--trainB_path',                dest='trainB_path',                 type=str,   default='./data/Air/',              help="Air images path")
parser.add_argument('--log_path',                   dest='log_path',                    type=str,   default='./data/UWGANckpt/logs/',          help="Log path" )
parser.add_argument('--ckpt_path',                  dest='ckpt_path',                   type=str,   default='./data/UWGANckpt/',   help="checkpoint path")
parser.add_argument('--test_path',                  dest='test_path',                   type=str,   default= './data/OceanDark/',          help="Test image path")
parser.add_argument('--gen_path',                   dest='gen_path',                    type=str,   default='./data/tempresult/',           help="Result path" )
# LL(Lowlight) arguments
parser.add_argument('--lowlight_images_path',       dest="lowlight_images_path",        type=str,   default="data/Dataset_Part1/",   help="LL training images path")
parser.add_argument('--lowlight_test_images_path',  dest="lowlight_test_images_path",   type=str,   default="./data/tempresult/",             help="LL test images path")
parser.add_argument('--lowlight_result_path',       dest="lowlight_result_path",        type=str,   default="./data/finalresult/",             help="LL Result images path")

parser.add_argument('--run_mode',            choices=[0, 1, 2, 3],             type=int, default=2,     help="Model Run mode to 0-Unet+clahe, 1-Unet+zdce, 2-Unet+clahe+zdce, 3-Unet+zdce+clahe")

parser.add_argument('--grad_clip_norm',             dest="grad_clip_norm",              type=float, default=0.1,                    help="Grad Clip")
parser.add_argument('--num_epochs_ll',              dest="num_epochs_ll",               type=int,   default=200,                    help="LL Epochs")
parser.add_argument('--train_batch_size_ll',        dest="train_batch_size_ll",         type=int,   default=8,                      help="LL training batch size")
parser.add_argument('--val_batch_size_ll',          dest="val_batch_size_ll",           type=int,   default=2,                      help="LL validation batch size")
# parser.add_argument('--num_workers_ll',             dest="num_workers_ll",              type=int,   default=4,                      help="No of worksers")
parser.add_argument('--display_iter_ll',            dest="display_iter_ll",             type=int,   default=2,                      help="Display iteration of LL")
parser.add_argument('--checkpoint_iter_ll',         dest="checkpoint_iter_ll",          type=int,   default=2,                      help="Checkpoints after to save iteration of LL")
parser.add_argument('--checkpoints_folder_ll',      dest="checkpoints_folder_ll",       type=str,   default="/data/weights/",             help="Checkpoints Folder LL")
parser.add_argument('--load_pretrain_ll',           action="store_true",           default=False,     help="Load Pretrained LL weights")
parser.add_argument('--pretrain_dir_ll',            dest="pretrain_dir_ll",             type=str,   default= "./data/weights/ep_169_it_62.h5",  help="Stored Weights for LL")

args = parser.parse_args()

def unet_train():
    # if args.use_gpu:
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    print("Params Config:\n")
    print("Learning Rate: %f" % args.lr)
    print("    Optimizer: Adam")
    print("   Batch Size: %d " % args.train_batch_size)
    print(" Train Epochs: %d " % args.num_epochs_ie)

    # rename pic for underwater image and ground truth image
    # BatchRename(image_path=trainA_path).rename()
    # BatchRename(image_path=trainB_path).rename()

    # underwater image
    image_u = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_u')

    # correct image
    image_r = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_r')

    training_flag = tf.compat.v1.placeholder(tf.bool)
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name='learning_rate')
    lr_sum = tf.compat.v1.summary.scalar('lr', learning_rate)

    # generated color image by u-net
    U_NET = UNet(input_=image_u, real_=image_r, is_training=training_flag)
    gen_image = U_NET.u_net(inputs=image_u, training=training_flag)
    G_sum = tf.compat.v1.summary.image("gen_image", gen_image, max_outputs=10)

    # loss of u-net
    errG = U_NET.l1_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.mse_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.ssim_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.msssim_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.gdl_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.l2_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)
    # errG = U_NET.ssim_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)
    # errG = U_NET.msssim_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)
    # errG = U_NET.gdl_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)

    errG_sum = tf.compat.v1.summary.scalar("loss", errG)
    t_var = tf.compat.v1.trainable_variables()
    g_vars = [var for var in t_var]

    # if consider l2 regularization
    # l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in t_var])

    # optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
    # train_op = optimizer.minimize(errG + l2_loss * weight_decay)
    train_op = optimizer.minimize(loss=errG)

    # TensorBoard Summaries
    # tf.summary.scalar('batch_loss', tf.reduce_mean(errG))
    # tf.summary.scalar('learning_rate', learning_rate)
    # try:
    #     tf.summary.scalar('l2_loss', tf.reduce_mean(l2_loss))
    # except: pass

    # saver = tf.train.Saver(tf.global_variables())

    config = tf.compat.v1.ConfigProto()
    # restrict model GPU memory utilization to min required
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:

        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)

        if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
            U_NET.restore(sess=sess, model_path=ckpt.model_checkpoint_path)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())

        all_sum = tf.compat.v1.summary.merge([G_sum, errG_sum, lr_sum])
        train_summary_writer = tf.compat.v1.summary.FileWriter(args.log_path, sess.graph)
        # merged_summary_op = tf.summary.merge_all()

        # load data
        # trainA_paths for underwater image
        # trainB_paths for ground truth image
        img_process = ImageProcess(pathA=args.trainA_path + '*.png',
                                   pathB=args.trainB_path + '*.png',
                                   batch_size=args.train_batch_size,
                                   is_aug=False)
        counter = 1
        trainA_paths, trainB_paths = img_process.load_data()
        for epoch in range(1, args.num_epochs_ie +1):
            # epoch_learning_rate = cosine_learning_rate(learn_rate=init_learning_rate,
            #                                            n_epochs=total_epochs,
            #                                            cur_epoch=epoch)
            epoch_learning_rate = args.lr
            # total_loss = []
            start_time = time.time()
            for step in range(1, int(len(trainA_paths)/args.train_batch_size)):

                batchA_images, batchB_images = img_process.shuffle_data(trainA_paths, trainB_paths)

                train_feed_dict = {
                    image_u: batchA_images,
                    image_r: batchB_images,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, summary_str = sess.run([train_op, all_sum], feed_dict=train_feed_dict)
                train_summary_writer.add_summary(summary=summary_str, global_step=counter)

                # batch_loss = sess.run(errG, feed_dict=train_feed_dict)
                # total_loss.append(batch_loss)
                counter += 1

            end_time = time.time()
            # train_loss = np.mean(total_loss)
            line = "epoch: %d/%d, time cost: %.4f\n" % (epoch, args.num_epochs_ie, float(end_time - start_time))
            # line = "epoch: %d/%d, train loss: %.4f, time cost: %.4f\n" % (epoch, total_epochs, float(train_loss), float(end_time - start_time))
            print(line)

            if epoch % 10 == 0:
                U_NET.save(sess=sess, model_path=args.ckpt_path + str(epoch)+'u_net.ckpt')

def unet_test():
    if not os.path.exists(args.gen_path):
        os.makedirs(args.gen_path)
    print("Params Config:\n")
    print("   Batch Size: %f" % args.test_batch_size)
    # test_batch_size)

    # underwater image
    image_u = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_u')
    # correct image
    image_r = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_r')

    # load model
    U_NET = UNet(input_=image_u, real_=image_r, is_training=False)
    gen_images = U_NET.u_net(inputs=image_u, training=False)

    # load image
    test_path = np.asarray(glob.glob(args.test_path + '*.jpg'))
    test_path.sort()
    # process image
    num_test_image = len(test_path)
    test_x = np.empty(shape=[num_test_image, 256, 256, 3], dtype=np.float32)
    i = 0
    for path_i in test_path:
        img_i = normalize_image(np.array(Image.open(path_i).resize(size=(256, 256))).astype(np.float32))
        # img_i = normalize_image(misc.imread(path_i).astype('float32'))
        test_x[i, :, :, :] = img_i
        i += 1
    # load checkpoints weight
    ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # compute FLOPs and params###################################################################
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(tf.compat.v1.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.compat.v1.profiler.profile(tf.compat.v1.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
        #############################################################################################

        U_NET.restore(sess, model_path=ckpt.model_checkpoint_path)

        test_pre_index = 0
        gen_num = 0
        gen_test_images = []
        begin_time = time.time()
        for test in range(int(num_test_image / args.test_batch_size)):
            test_batch_x = test_x[test_pre_index:test_pre_index+args.test_batch_size]
            test_pre_index = test_pre_index + args.test_batch_size

            test_feed_dict = {image_u: test_batch_x}

            gen_test = sess.run(gen_images, feed_dict=test_feed_dict)
            # print(gen_test.max())
            # print(gen_test.min())
            # print(gen_test.shape)
            gen_test_images.append(gen_test)
            # print(len(gen_test_images))
            # print(gen_test_images[0][0].shape)
        end_time = time.time()

        print("Total Test Time: %.4f, Average Time Per Image: %.6f, FPS: %.4f" %
              (end_time - begin_time,
               (end_time - begin_time) / num_test_image,
               num_test_image / (end_time - begin_time)))

        for img_g in gen_test_images:
            # misc.imsave(gen_path + str(gen_num) + '_gen.png', img_g[0])
            #misc.imsave(gen_path + test_path[gen_num].split('/')[-1][:-4] + '_gen.png', img_g[0])
            # temp = test_path[gen_num].split('/')[2].split('\\')[1] + '_gen.png'
            temp =  os.path.basename(test_path[gen_num]).split('.')[0] + '_gen.png'
            imageio.imwrite(args.gen_path + temp, img_g[0])
            # print(img_g[0].shape)
            gen_num += 1
        print("Done with test image, gen image: %d" % gen_num)

def zeroDCETrain():
    path = os.getcwd() + args.checkpoints_folder_ll
    if not os.path.exists(path):
        os.mkdir(path)
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)


    #tf.config.gpu.set_per_process_memory_fraction(0.75)
    #tf.config.gpu.set_per_process_memory_growth(True)    

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

    train_dataset = DataGenerator(args.lowlight_images_path, args.train_batch_size_ll)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model =  zeroDECModel()
    
    beginepoch = 0
    # Load model to resume training
    if(args.load_pretrain_ll):
      checkpointepoch =  re.search(r'\d+', ntpath.basename(args.pretrain_dir_ll)).group(0) 
      print("last epoch on"+checkpointepoch)   
      beginepoch = int(checkpointepoch)
      model.load_weights(args.pretrain_dir_ll)

    min_loss = 10000.0
    print("Start training ...")
    for epoch in range(beginepoch, args.num_epochs_ll):
        for iteration, img_lowlight in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                A = model(img_lowlight)
                r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
                x = img_lowlight + r1 * (tf.pow(img_lowlight,2)-img_lowlight)
                x = x + r2 * (tf.pow(x,2)-x)
                x = x + r3 * (tf.pow(x,2)-x)
                enhanced_image_1 = x + r4*(tf.pow(x,2)-x)
                x = enhanced_image_1 + r5*(tf.pow(enhanced_image_1,2)-enhanced_image_1)		
                x = x + r6*(tf.pow(x,2)-x)	
                x = x + r7*(tf.pow(x,2)-x)
                enhance_image = x + r8*(tf.pow(x,2)-x)
                
                loss_TV = 200*L_TV(A)
                loss_spa = tf.reduce_mean(L_spa(enhance_image, img_lowlight))
                loss_col = 5*tf.reduce_mean(L_color(enhance_image))
                loss_exp = 10*tf.reduce_mean(L_exp(enhance_image, mean_val=0.6))

                total_loss = loss_TV + loss_spa + loss_col + loss_exp

            grads = tape.gradient(total_loss, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # if iteration % config.display_iter == 0:
            #     print("Training loss (for one batch) at step %d: %.4f" % (iteration, float(total_loss)))

            progress(epoch+1, (iteration+1), len(train_dataset), total_loss=total_loss)

            if (iteration+1) % args.checkpoint_iter_ll == 0 and total_loss < min_loss:
                min_loss = total_loss
                progress(epoch+1, (iteration+1), len(train_dataset), total_loss=total_loss, message=' ----- saved weight for epoch ' + str(epoch+1) + ' iter ' + str(iteration+1))
                model.save_weights(os.path.join(os.getcwd() + args.checkpoints_folder_ll, "ep_"+str(epoch+1)+"_it_"+str(iteration+1)+".h5"))

def zeroDCETest():
    if not os.path.exists(args.lowlight_result_path):
        os.makedirs(args.lowlight_result_path)

    model =  zeroDECModel()

    model.load_weights("data/weights/best.h5")

    ### load image ###
    for test_file in glob.glob(args.lowlight_test_images_path + "*.*"):
        data_lowlight_path = test_file
        original_img = Image.open(data_lowlight_path)
        original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])

        original_img = original_img.resize((256,256), Image.ANTIALIAS) 
        original_img = (np.asarray(original_img)/255.0)

        img_lowlight = Image.open(data_lowlight_path)
                
        img_lowlight = img_lowlight.resize((256,256), Image.ANTIALIAS)

        img_lowlight = (np.asarray(img_lowlight)/255.0) 
        img_lowlight = np.expand_dims(img_lowlight, 0)
        # img_lowlight = K.constant(img_lowlight)

        if(args.run_mode == 2):
            img_lowlight = tf_clahe.clahe(img_lowlight)

        ### process image ###
        A = model.predict(img_lowlight)
        r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
        x = original_img + r1 * (K.pow(original_img,2)-original_img)
        x = x + r2 * (K.pow(x,2)-x)
        x = x + r3 * (K.pow(x,2)-x)
        enhanced_image_1 = x + r4*(K.pow(x,2)-x)
        x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
        x = x + r6*(K.pow(x,2)-x)	
        x = x + r7*(K.pow(x,2)-x)
        enhance_image = x + r8*(K.pow(x,2)-x)
        enhance_image = tf.cast((enhance_image[0,:,:,:] * 255), dtype=np.uint8)
        
        filename = os.getcwd() + args.lowlight_result_path + os.path.basename(test_file)

        fileExt = "_rs.png"
        if(args.run_mode == 3):
            enhance_image_temp = Image.fromarray(enhance_image.numpy())
            # enhance_image_temp = enhance_image_temp.resize(original_size, Image.ANTIALIAS)
            fileExt = "_unet_zdce_preclahe.png"
            enhance_image_temp.save(filename.replace(".png", fileExt))

        if(args.run_mode == 3):
            enhance_image = tf_clahe.clahe(enhance_image)

        enhance_image = Image.fromarray(enhance_image.numpy())
        # enhance_image = enhance_image.resize(original_size, Image.ANTIALIAS)
        filename = os.getcwd() + args.lowlight_result_path + os.path.basename(test_file)

        fileExt = "_rs.png"
        if(args.run_mode == 1):
            fileExt = "_unet_zdce.png"
        elif(args.run_mode == 2):
            fileExt = "_unet_clahe_zdce.png"
        elif(args.run_mode == 3):
            fileExt = "_unet_zdce_clahe.png"

        enhance_image.save(filename.replace(".png", fileExt))
        
    print("Saved generated image")

def clahe():
    for test_file in glob.glob(args.lowlight_test_images_path + "*.png"):
        data_lowlight_path = test_file
        original_img = Image.open(data_lowlight_path)
        original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])

        img_lowlight = tf.io.decode_image(tf.io.read_file(data_lowlight_path))
        enhance_image = tf_clahe.clahe(img_lowlight)

        enhance_image = Image.fromarray(enhance_image.numpy())
        # enhance_image = enhance_image.resize(original_size, Image.ANTIALIAS)
        filename = os.getcwd() + args.lowlight_result_path + os.path.basename(test_file)
        enhance_image.save(filename.replace(".png", "_unet_clahe.png"))
        
    print("Saved generated image")

def EnhanceImage():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.phase == 'train':
            unet_train()
        elif args.phase == 'test':
            unet_test()
        else:
            print('[!] Unknown phase')
            exit(0)

        sess.close()

def EnhanceLowLight():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    print("Begining LL")
    if args.phase == 'train':
        zeroDCETrain()
        # lowlight_train(model)
    elif args.phase == 'test':
        zeroDCETest()

def main(_):
    print("[*] GPU\n")
    print("Phase : " + str(args.phase) )
    print("Run Mode : " + str(args.run_mode) )
    print("LL Train resume : " + str(args.load_pretrain_ll) )
    process_train = multiprocessing.Process(target=EnhanceImage)
    process_train.start()
    process_train.join()

    if(args.run_mode == 0):
        clahe()

    if(args.run_mode > 0):
        process_train = multiprocessing.Process(target=EnhanceLowLight)
        process_train.start()
        process_train.join()

if __name__ == "__main__":
    tf.compat.v1.app.run()