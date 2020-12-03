#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

import denoiseg
from denoiseg.models import DenoiSeg, DenoiSegConfig
from denoiseg.utils.misc_utils import combine_train_test_data, shuffle_train_data, augment_data
from denoiseg.utils.seg_utils import *
from denoiseg.utils.compute_precision_threshold import measure_precision

from csbdeep.utils import plot_history

import tensorflow as tf
import keras.backend as K

import urllib
import os
import zipfile

def rename_tensor(name_in, name_out):
    print("  " + name_out + " (" + name_in + ")")
    var = tf.get_default_graph().get_tensor_by_name(name_in)
    tf.identity(var, name=name_out)


def rename_op(name_in, name_out):
    print("  " + name_out + " (" + name_in + ")")
    op = tf.get_default_graph().get_operation_by_name(name_in)
    tf.group(op, name=name_out)


def get_name(tensor):
    return ''.join(tensor.name.rsplit(":0", 1))

X = np.random.uniform(-2,2,(10, 96, 96, 1))
X_val = np.random.uniform(-2,2,(10, 96, 96, 1))
Y = np.random.uniform(-2,2,(10, 96, 96, 3))
Y_val = np.random.uniform(-2,2,(10, 96, 96, 3))

train_batch_size = 128
train_steps_per_epoch = max(100, min(int(X.shape[0]/train_batch_size), 400))

conf = DenoiSegConfig(X, unet_kern_size=3, n_channel_out=4, relative_weights = [1.0,1.0,5.0],
                      train_steps_per_epoch=train_steps_per_epoch, train_epochs=200, 
                      batch_norm=True, train_batch_size=128, unet_n_first = 32, 
                      unet_n_depth=4, denoiseg_alpha=0.5, train_tensorboard=False)

vars(conf)

model_name = 'denoiseg_model'
basedir = 'models'
model = DenoiSeg(conf, model_name, basedir)

# this is the general CSBDeep TensorFlow graph builder part which might work for all CSBDeep models

model.prepare_for_training()
model.keras_model._make_train_function()
model.keras_model._make_test_function()
model.keras_model._make_predict_function()
# model.keras_model.summary()

print("\nAvailable inputs: ")
for i in range(len(model.keras_model.inputs)):
    target = model.keras_model.inputs[i]
    print("  " + get_name(target))

print("\nAvailable loss tensors: ")
for i in range(len(model.keras_model.metrics_names)):
    loss_name = model.keras_model.metrics_names[i] + "_tensor"
    if i is 0:
        rename_tensor(model.keras_model.total_loss.name, loss_name)
    else:
        rename_tensor(model.keras_model.metrics_tensors[i-1].name, loss_name)

print("\nAvailable training targets: ")
for i in range(len(model.keras_model.targets)):
    target = model.keras_model.targets[i]
    print("  " + get_name(target))

print("\nAvailable inference outputs: ")
for i in range(len(model.keras_model.outputs)):
    target = model.keras_model.outputs[i]
    name = model.keras_model.output_names[i]
    rename_tensor(target.name, name + "_tensor")

print("\nAvailable learning rate tensors: ")
learning_rate_tensor_name = get_name(model.keras_model.optimizer.lr)
rename_tensor(learning_rate_tensor_name + "/read:0", "read_learning_rate")
rename_tensor(learning_rate_tensor_name + ":0", "write_learning_rate")

print("\nAvailable target operations:")
rename_op('training/group_deps', 'train')
rename_op("group_deps", "validation")

# This part is DenoiSeg specific (just improving the output tensor names a bit):
print("\nOutput name aliases:")
rename_tensor('out_segment_tensor:0', 'segmented')
rename_tensor('out_denoise_tensor:0', 'denoised')


print("\nDenoiSeg version " + denoiseg.__version__)


tf.global_variables_initializer()
tf.train.Saver().as_saver_def()

# save graph for training
with open('../resources/denoiseg_graph_2d.pb', 'wb') as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())

# [n.name for n in tf.get_default_graph().as_graph_def().node]


# save frozen graph for prediction
#import shutil
#shutil.rmtree("prediction_2d")
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    tf.tables_initializer().run()
    #print(sess.graph.get_tensor_by_name('keras_learning_phase/input:0'))
    #sess.run(tf.variables_initializer([tf.compat.v1.get_variable("conv2d_1/bias:0")]))
    #sess.run(init)
    sess.run("init")
    model_input = tf.compat.v1.saved_model.build_tensor_info(sess.graph.get_tensor_by_name('input:0'))
    model_output_denoise = tf.compat.v1.saved_model.build_tensor_info(sess.graph.get_tensor_by_name('denoised:0'))
    model_output_segment = tf.compat.v1.saved_model.build_tensor_info(sess.graph.get_tensor_by_name('segmented:0'))

    signature_definition = tf.compat.v1.saved_model.build_signature_def(
        inputs={"input": model_input},
        outputs={"denoised": model_output_denoise, "segmented": model_output_segment},
        method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder = tf.compat.v1.saved_model.Builder("../resources/denoiseg_prediction_2d")
    builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], signature_def_map={
        tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
    }, clear_devices=True)
    builder.save()
