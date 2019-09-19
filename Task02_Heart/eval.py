import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import model
import mriHandler
import utils
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Segment unseen heart MRIs")
parser.add_argument("-c", "--checkpoint", type=str, help="path to checkpoint/weights", default="checkpoints/model-3800")
parser.add_argument("-i", "--img_no", type=int, help="Path to MRI", default=0)
args = parser.parse_args()

checkpoint = args.checkpoint
img_no = args.img_no

dh = mriHandler.MRIHandler()
INPUT_SHAPE = dh.INPUT_SHAPE

input_im = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], 1], name="input")

segnet = model.SegNetBasic(2)
prediction = segnet.predict(input_im, False)
saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer()
    saver.restore(sess, checkpoint)
    #img, label = dh.load_val_data("val/")
    img, _, label = dh.next_batch(3)
    pred = sess.run(prediction, feed_dict={input_im: np.expand_dims(img[img_no],0)})
    pred = utils.create_prediction_and_label(pred)

utils.show_prediction_and_label(pred, label[img_no], img)