# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================

import os

import numpy as np
import tensorflow as tf
import cv2

from datasets import datasets_factory
from preprocessing import preprocessing_factory
from nets import nets_factory
from lib.nms.py_cpu_nms import py_cpu_nms


slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(_):

    batch_size = 32
    dataset = datasets_factory.get_dataset('pascalvoc_2007', 'train', 'tfrecords')

    with tf.name_scope('pascalvoc_2007_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=1,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 * batch_size,
                    shuffle=False)

    [image_org, glabels, gbboxes, ss] = provider.get(['image', 'object/label',
                                                      'object/bbox', 'image/ss'])

    # Preprocessing image, label and other infos
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                                    'vgg',
                                    is_training=True)
    glabels = tf.expand_dims(glabels, 0)
    ss_reshape = tf.reshape(ss, [200, 5])

    # Backbone network
    image = image_preprocessing_fn(image_org, out_shape=(224, 224))
    vgg_16 = nets_factory.get_network_fn('vgg_16', num_classes=20)
    fc8c, fc8d, _ = vgg_16(image, spatial_squeeze=False, rois=ss_reshape)


    # Network output delta
    delta_fc8c = tf.nn.softmax(fc8c, dim=-1)
    delta_fc8d = tf.nn.softmax(fc8d, dim=0)

    # Region score
    xr = tf.multiply(delta_fc8c, delta_fc8d)

    # Classification score
    yc = tf.squeeze(tf.reduce_sum(xr, 0), 1)

    # Loss
    loss_ = tf.multiply(glabels, yc-0.5)+0.5
    loss = - tf.reduce_sum( tf.log( loss_ + 1e-8 ))

    train_op = tf.train.MomentumOptimizer(learning_rate=1e-6, momentum=0.9).minimize(loss)

    # Restore variables
    variables_to_restore = []
    #ignore_variables = ['vgg_16/fc8', 'vgg_16/fc7', 'vgg_16/fc6']
    ignore_variables = []
    for var in slim.get_model_variables():
        need_to_restore = True
        for igvar in ignore_variables:
            if var.op.name.startswith(igvar):
                need_to_restore = False
        if need_to_restore:
            variables_to_restore.append(var)

    #checkpoint_path = './checkpoints/vgg_16.ckpt'
    checkpoint_path = './checkpoints/ex1/model.ckpt-65000'
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path,
                                          variables_to_restore,
                                          ignore_missing_vars=False)


    #saver = tf.train.Saver(slim.get_model_variables())

    #summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            # Run initilizing
            sess.run(tf.global_variables_initializer())
            sess.run(init_assign_op, init_feed_dict)

            # Session's summary
            #summary_writer = tf.summary.FileWriter('logs', sess.graph)

            # Restore checkpoints

            # Train
            '''
            for i in range(10000 * 10):
                _, output_loss = sess.run([train_op, loss])

                if i % 100 == 0:
                    print(i, output_loss, flush=True)


                if i % 5000 == 0:
                    saver.save(sess, 'checkpoints/ex1/model.ckpt', \
                            global_step = i, write_meta_graph=False, \
                            write_state=False)
            '''
            # Train eval
            for i in range(200):
                _, output_loss, output_xr, output_rois, output_image = sess.run([train_op, loss,\
                     xr,  ss_reshape, image_org])
                print(i, output_loss)
                output_rois = output_rois[:,1:]
                output_xr = np.squeeze(output_xr)
                output_image = output_image[:,:,::-1]

                cv2.imwrite('tmp/vis'+str(i)+'.jpg', output_image)
                imOut = cv2.imread('tmp/vis'+str(i)+'.jpg')
                threshold = 1e-3
                for c in range(20):
                    scores = output_xr[:,c].reshape((200,1))
                    dets = np.hstack((output_rois, scores))
                    keep_inds = py_cpu_nms(dets, 0.4)
                    for ind in keep_inds:
                        bbox = output_rois[ind]
                        score = scores[ind]
                        if score > threshold:
                             cv2.rectangle(imOut, (int(bbox[0]+1), int(bbox[1]+1)), \
                                (int(bbox[2]-1), int(bbox[3]-1)), (0, 255, 0), 1, cv2.LINE_AA)

                cv2.imwrite('tmp/vis' + str(i) + '.jpg', imOut)


if __name__ == '__main__':
    tf.app.run()
