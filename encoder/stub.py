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

    for dropout in ga_data['drop_dropouts']
        subgraph = tf.contrib.graph_editor.make_view_from_scope(dropout, graph)
        sgv_input = subgraph.inputs[0]
        tf.contrib.graph_editor.detach_inputs(sgv_input)
        reroute.reroute_ts([sgv_input], list(sgv.outputs)[0])

    
    # all_ops = tf.contrib.graph_editor.get_forward_walk_ops(images, stop_at_ts=(logits['fcn_logits']))
    # conv_ops = tf.contrib.graph_editor.filter_ops(all_ops, lambda op: op.type == "Conv2D")
    # dropout_ops = tf.contrib.graph_editor.filter_ops(all_ops, lambda op: op.type == "Dropout")
    # print(conv_ops)
    # print(dropout_ops)
