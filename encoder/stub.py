"""Stub for GA training."""

import os
import imp
import commentjson

import tensorflow as tf


def inference(hypes, images, train=True):
    with open(hypes['ga_data']) as f:
        ga_data = commentjson.load(f)

    encoder_path = os.path.join(os.path.dirname(__file__), 
                                os.path.basename(ga_data['encoder_path']))
    encoder = imp.load_source(ga_data['encoder_name'], encoder_path)
    logits = encoder.inference(hypes, images, train=True)
    graph = tf.get_default_graph()

    for layer in ga_data['drop']:
        subgraph = tf.contrib.graph_editor.make_view_from_scope(layer, graph)
        sgv_input = list(subgraph.inputs)[0]

        if 'dropout' in layer: 
            subgraph, _ = tf.contrib.graph_editor.bypass(subgraph)
        elif layer in ['conv4_1', 'conv3_1', 'conv2_1']:
            tf.contrib.graph_editor.detach_inputs(subgraph)
            tile = tf.tile(sgv_input, multiples=[1, 1, 1, 2])
            sgv_output = graph.get_tensor_by_name('{}/Relu:0'.format(layer))
            tf.contrib.graph_editor.reroute_ts([tile], [sgv_output])
        elif 'conv' in layer or 'fc7' in layer:
            tf.contrib.graph_editor.detach_inputs(subgraph)
            sgv_output = graph.get_tensor_by_name('{}/Relu:0'.format(layer))
            tf.contrib.graph_editor.reroute_ts([sgv_input], [sgv_output])
        elif 'fc6' in layer:
            tf.contrib.graph_editor.detach_inputs(subgraph)
            tile = tf.tile(sgv_input, multiples=[1, 1, 1, 8])
            sgv_output = graph.get_tensor_by_name('{}/Relu:0'.format(layer))
            tf.contrib.graph_editor.reroute_ts([tile], [sgv_output])
        elif 'score_pool3' in layer:
            add = graph.get_operation_by_name('Add_1')
            sgv_input= graph.get_tensor_by_name('upscore4/conv2d_transpose:0')
            sgv_output = list(add.outputs)[0]
            tf.contrib.graph_editor.detach_inputs(add)
            tf.contrib.graph_editor.reroute_ts([sgv_input], [sgv_output])
        elif 'score_pool4' in layer:
            add = graph.get_operation_by_name('Add')
            sgv_input= graph.get_tensor_by_name('upscore2/conv2d_transpose:0')
            sgv_output = list(add.outputs)[0]
            tf.contrib.graph_editor.detach_inputs(add)
            tf.contrib.graph_editor.reroute_ts([sgv_input], [sgv_output])
 
    return logits 
