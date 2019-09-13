import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")  # noqa
import model
import mriHandler
import utils
import argparse
tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description="Which training step")
parser.add_argument("-c", "--checkpoint", type=int, default=None, help="Enter the checkpoint number")
parser = parser.parse_args()
STEP = parser.checkpoint

# Data Handler
dh = mriHandler.MRIHandler()
INPUT_SHAPE = dh.INPUT_SHAPE
MAX_CLASS = dh.classes

# Input MRI and labeled Mask
input_im = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], 1],
                          name="input")
mask = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], MAX_CLASS],
                      name="Segmentation")

# Placeholders for summaries
loss_feed = tf.placeholder(tf.float32)
pixel_acc_feed = tf.placeholder(tf.float32)
val_pixel_acc_feed = tf.placeholder(tf.float32)

# Summaries
loss_summary = tf.summary.scalar("Loss", loss_feed)
pixel_acc_summary = tf.summary.scalar("Pixel_Acc", pixel_acc_feed)
val_pixel_acc_summary = tf.summary.scalar("Val_Pixel_Acc", val_pixel_acc_feed)


# OPs from model
segnet = model.SegNetBasic(2)
prediction = segnet.predict(input_im, is_training=True)
cost = segnet.loss(prediction, mask)
optimizer = segnet.optimizer(1e-5)
train_op = optimizer.minimize(cost)

# Logwriter
saver = tf.train.Saver(max_to_keep=30)
logdir = "logs/training"
file_writer = tf.summary.FileWriter(logdir, flush_secs=5)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if STEP:
        saver.restore(sess, f'checkpoints/model-{STEP}')

    for i in range(STEP, 6000):

        img_batch, label_batch, orig_labels = dh.next_batch(1)
        _, loss, pred = sess.run([train_op, cost, prediction], feed_dict={input_im: img_batch, mask: label_batch})

        predicted_segmentation = utils.create_prediction_and_label(pred)
        pixel_acc = utils.pixel_accuracy(predicted_segmentation, orig_labels, MAX_CLASS)
        print(f'Iteration:{i} Loss:{loss} Pixel Accuracy: {pixel_acc}')

        summaries = [loss_summary, pixel_acc_summary]
        summary_op = tf.summary.merge(summaries)
        summary = sess.run(summary_op, feed_dict={loss_feed: loss, pixel_acc_feed: pixel_acc})
        file_writer.add_summary(summary, i)

        if i % 10 == 0:
            img, label = dh.load_val_data("val/")
            pred = sess.run(prediction, feed_dict={input_im: np.expand_dims(img[0], 0)})
            pred = utils.create_prediction_and_label(pred)
            pred_sum = np.sum(pred)
            pixel_acc = utils.pixel_accuracy(pred, label, MAX_CLASS)
            summary = sess.run(val_pixel_acc_summary, feed_dict={val_pixel_acc_feed: pixel_acc})
            file_writer.add_summary(summary, i)
            print(f'Ones in Pred:{pred_sum} Val Pixel Acc:{pixel_acc}')

        if i % 100 == 0 and i > 0:
            saver.save(sess, "checkpoints/model", global_step=i)

