import os
import pprint
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from uwgan_model import UWGAN


pp = pprint.PrettyPrinter()

flags = tf.compat.v1.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64], test size is 100")
flags.DEFINE_integer("input_height", 480, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 640, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 256, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of im  age color. [3]")
flags.DEFINE_string("water_dataset", "Type_3", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("air_dataset", "air_images", "The name of dataset with air images")
flags.DEFINE_string("depth_dataset", "air_depth", "The name of dataset with depth images")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_color3", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dirc", "logs_color3", "Directory name to save the logs [logs]")
flags.DEFINE_string("results_dir", "results_color3", "Directory name to save the checkpoints [results]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("save_epoch", 5, "The size of the output images to produce. If None, same value as output_height [None]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.results_dir):
        os.mkdir(FLAGS.results_dir)
    if not os.path.exists(FLAGS.log_dirc):
        os.mkdir(FLAGS.log_dirc)

    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=run_config) as sess:
        wgan = UWGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            c_dim=FLAGS.c_dim,
            save_epoch=FLAGS.save_epoch,
            water_dataset_name=FLAGS.water_dataset,
            air_dataset_name = FLAGS.air_dataset,
            depth_dataset_name = FLAGS.depth_dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            is_crop=FLAGS.is_crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            results_dir = FLAGS.results_dir)

        if FLAGS.is_train:
            wgan.train(FLAGS)
        else:
            if not wgan.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")
            wgan.test(FLAGS)


if __name__ == '__main__':

    tf.compat.v1.app.run()
