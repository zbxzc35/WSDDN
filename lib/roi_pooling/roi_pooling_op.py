import tensorflow as tf
from tensorflow.python.framework import ops
import os

filename = os.path.join(os.path.dirname(__file__), 'roi_pooling.so')
_roi_pooling_module = tf.load_op_library(filename)
roi_pool = _roi_pooling_module.roi_pool
roi_pool_grad = _roi_pooling_module.roi_pool_grad


@ops.RegisterShape('RoiPool')
def _roi_pool_shape(op):
    '''
    '''
    dims_data = op.inputs[0].get_shape().as_list()
    channels = dims_data[3]
    dims_rois = op.inputs[1].get_shape().as_list()
    num_rois = dims_rois[0]

    pooled_height = op.get_attr('pooled_height')
    pooled_width = op.get_attr('pooled_width')

    output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, channels])
    return [output_shape, output_shape]


@ops.RegisterGradient('RoiPool')
def _roi_pool_grad(op, grad, _):
    '''
    '''
    data = op.inputs[0]
    rois = op.inputs[1]
    argmax = op.outputs[1]
    pooled_height = op.get_attr('pooled_height')
    pooled_width = op.get_attr('pooled_width')
    spatial_scale = op.get_attr('spatial_scale')

    # compute gradient
    data_grad = _roi_pooling_module.roi_pool_grad(data, rois, argmax, grad,
                                             pooled_height, pooled_width, spatial_scale)

    return [data_grad, None]  # List of one Tensor, since we have one input
