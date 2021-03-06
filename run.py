from datetime import datetime
import time
import h5py
import numpy as np
import tensorflow as tf
from utils import *
from AlexNet import Alexnet
import sys
import os

now = datetime.now()
logs_path = "./graph/" + now.strftime("%Y%m%d-%H%M%S")
save_dir = './checkpoints/'

image_size = 256
num_classes = 22
num_channels = 1
num_epochs = 100
batch_size = 32
display = 100


h5f = h5py.File('chest_Xray_data.h5', 'r')
X_train = h5f['X_train'][:]
Y_train = h5f['Y_train'][:]
h5f.close()

X_train, _ = reformat(X_train, Y_train, image_size, num_channels, num_classes)
Y_train = np.squeeze(Y_train)


# Creating the alexnet model
model = Alexnet(num_classes, image_size, num_channels, batch_size)
model.inference().pred_func().accuracy_func().loss_func().train_func()

# Saving the best trained model (based on the validation accuracy)
saver = tf.train.Saver()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

avg_acc_b_all = mean_acc = loss_b_all = mean_loss = epoch_acc = epoch_loss = np.array([])
acc_b_all = np.zeros((22))
sum_count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized")
    merged = tf.summary.merge_all()
    batch_writer = tf.summary.FileWriter(logs_path + '/batch/', sess.graph)
    valid_writer = tf.summary.FileWriter(logs_path + '/valid/')
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        acc_val = loss_val = np.array([])
        y_prob_val = np.zeros((0, 2))
        print('-----------------------------------------------------------------------------')
        print('Epoch: {}'.format(epoch+1))
        X_train, Y_train = randomize(X_train, Y_train)
        step_count = int(len(X_train)/batch_size)
        for step in range(step_count):
            start = step * batch_size
            end = (step + 1) * batch_size
            X_batch, Y_batch = get_next_batch(X_train, Y_train, start, end)
            # X_batch = random_rotation_2d(X_batch, 20.0)
            feed_dict_batch = {model.x: X_batch, model.y: Y_batch, model.keep_prob: 0.5}

            _, avg_acc_b, acc_b, loss_b, y_pred = sess.run([model.train_op,
                                                        model.avg_accuracy,
                                                        model.accuracy,
                                                        model.loss,
                                                        model.probs
                                                        ], feed_dict=feed_dict_batch)
            acc_b_all = np.concatenate((acc_b_all, acc_b))
            avg_acc_b_all = np.append(avg_acc_b_all, avg_acc_b)
            loss_b_all = np.append(loss_b_all, loss_b)

            if step % display == 0:
                mean_acc = np.mean(avg_acc_b_all)
                mean_loss = np.mean(loss_b_all)
                print('step %i, training loss: %.5f, training accuracy: %.3f' % (step, mean_loss, mean_acc*100))
                summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Mean_Accuracy', simple_value=mean_acc*100)])
                batch_writer.add_summary(summary_tr, sum_count * display)
                summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Mean_Loss', simple_value=mean_loss)])
                batch_writer.add_summary(summary_tr, sum_count * display)
                summary = sess.run(merged, feed_dict=feed_dict_batch)
                batch_writer.add_summary(summary, sum_count * display)
                sum_count += 1
                avg_acc_b_all = loss_b_all = mean_acc = mean_loss = np.array([])
                acc_Cond_all = acc_b_all.reshape(-1, num_classes)[1:, :]
                acc_Cond = np.mean(acc_Cond_all,axis=0)
                print(acc_Cond * 100)
                acc_b_all = np.zeros((22))
