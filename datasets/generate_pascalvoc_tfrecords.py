# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' Generate tfrecords for PASCAL VOC datasets
Modification from https://github.com/balancap/SSD-Tensorflow/blob/master/datasets/pascalvoc_to_tfrecords.py

Each record contains following fields:

    image/encoded: raw image encoded in RGB
    image/height, image/width, image/channel: height, width, channel
    image/object/bbox/xmin, xmax, ymin, ymax: list of bboxes coords
    image/object/label: list of integer specifying the classification label for each box

'''
import os
import sys
import random

import numpy as np
import tensorflow as tf
import cv2

import xml.etree.ElementTree as ET

from dataset_utils import int64_feature, float_feature, bytes_feature

# VOC labels
VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

def _process_image(directory, name):
    '''
    '''

    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    # Find annotations.
    bboxes = []
    labels = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))


    # preprocess labels
    f_labels = [-1 for i in range(20)]
    for label in labels:
        # label - 1: ignore background
        f_labels[label-1] = 1

    return image_data, shape, bboxes, f_labels, difficult, truncated

def  _convert_to_example(image_data, shape, bboxes, labels,
                        difficult, truncated, ss_data):
    '''
    '''
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': float_feature(labels),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/selectivesearch':  float_feature(ss_data.tolist()),
            'image/encoded': bytes_feature(image_data)}))
    return example

def _get_selective_search(directory, name):
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    org = cv2.imread(filename)

    height, width = org.shape[0], org.shape[1]
    _height, _width = height, width

    if height > 2*width:
        _width = height * 0.5
    if width > 2*height:
        _height = width *0.5

    im = cv2.resize(org, (int(_width), int(_height)))

    n_bboxes = 200
    results = np.zeros((n_bboxes*5))

    # initilize cv2 to accelerate
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()

    rects = np.array(ss.process())

    # Visualization
    imOut = org.copy()
    scale_h, scale_w = _height / height, _width / width
    for i, rect in enumerate(rects):
        if i < n_bboxes:
            x, y, w, h = rect
            xmin, ymin, xmax, ymax = int(x/scale_w), int(y/scale_h), \
                                    int((x+w)/scale_w), int((y+h)/scale_h)

            results[i*5:i*5+5] = 0, xmin, ymin, xmax, ymax

            cv2.rectangle(imOut, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite('tmp/vis' + name + '.jpg', imOut)

    return results



def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, difficult, truncated = \
        _process_image(dataset_dir, name)

    # add selective search data
    ss_data = _get_selective_search(dataset_dir, name)

    example = _convert_to_example(image_data, shape, bboxes, labels,
                                difficult, truncated, ss_data)
    tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)


    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    print('\nFinished converting the Pascal VOC dataset!')

if __name__ =='__main__':
    '''
    usage: ./generate_pascalvoc_tfrecords.py /path/to/VOCdevkit/VOC2007 /path/to/output/
    '''
    DATASET_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    run(DATASET_DIR, OUTPUT_DIR)
