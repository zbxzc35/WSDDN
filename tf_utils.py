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
# ==============================================================================
"""Diverse TensorFlow utils, for training, evaluation and so on!
"""
import tensorflow as tf

slim = tf.contrib.slim


# =========================================================================== #
# General tools.
# =========================================================================== #
def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.

    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

def pad_list_fixed_size(l, size=20):
    r = []

    n = tf.shape(l)[0]
    if len(l.get_shape()) == 1:
        paddings = [[0, 0], [0, size-n]]
        r = tf.reshape(l, [n, 1])
        r = tf.pad(r, paddings, 'CONSTANT', constant_values=-1)
        r = tf.reshape(r, [size])
    elif len(l.get_shape()) == 2:
        paddings = [[0, 0], [0, size-n]]
        r = tf.pad(l, paddings, 'CONSTANT', constant_values=-1)
        r = tf.reshape(r, [size, 4])

    return r

