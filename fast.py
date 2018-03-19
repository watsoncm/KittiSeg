"""
Uses a series of hyperparameter tuning methods in order to speed up FCN8

Based on train.py by Marvin Teichmann
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import commentjson
import logging
import os
import sys
import string
import numpy as np

import collections


NUM_GENERATIONS = 10
MEMBERS_PER_GEN = 5
layerList = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
             'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2',
             'conv5_3', 'fc6', 'fc7']
layerPrior = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3,
              0.4, 0.5, 0.4, 0.3, 0.3, 0.2,
              0.2, 0.1, 0.1]

fastDict = {'drop_layers': []}

# configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
		level=logging.INFO,
		stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.utils as utils

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', None,
                    'File storing model parameters.')

flags.DEFINE_string('mod', None,
                    'Modifier for model parameters.')

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))


def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    if tf.app.flags.FLAGS.hypes is None:
        logging.error("No hype file is given.")
        logging.info("Usage: python train.py --hypes hypes/KittiClass.json")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = commentjson.load(f)
    utils.load_plugins()

    if tf.app.flags.FLAGS.mod is not None:
        import ast
        mod_dict = ast.literal_eval(tf.app.flags.FLAGS.mod)
        dict_merge(hypes, mod_dict)

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'KittiSeg')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    train.maybe_download_and_extract(hypes)
    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    logging.info("Start training")
  
    #######

    ga_content = {'encoder_name': 'fcn8_vgg',
                  'encoder_path': hypes['model']['architecture_file'], 
                  'drop': ['conv4_1']}
    hypes['model']['architecture_file'] = '../encoder/stub.py'
    hypes['ga_data'] = 'ga_data.json'

    with open(hypes['ga_data'], 'w') as f:
        commentjson.dump(ga_content, f)
   
    train.do_training(hypes)
    

# pass on to next generation??

    """
    prevTopDropLists = []
    for i in NUM_GENERATIONS:
        childrenRuntime = np.zeros(MEMBERS_PER_GEN)
        childrenDropLists = []
        prevListsCombined = combineDropLists(prevTopDropLists, len(layerList))
        for j in range(MEMBERS_PER_GEN): # for each child
            dropList = []
            if j!=0 or j!=1: #first should be the control/empty droplist per generation, second is hybrid of above
                for k in range(len(layerList)):
                    if layerPrior[k] >= np.random.random():
                        dropList.append(layerList[k])
            if j==1:
                dropList = prevListsCombined
            with open(hypes['ga_data'], 'w') as f:
                commentjson.dump(ga_content, f)
            ga_content['drop'] = dropList
            train.do_training(hypes)
            childrenRuntime[j] = 5 #Chandler put actual runtime here pls
            childrenDropLists.append(dropList)
        # pull out top n children's droplists
        sortedChildrenRuntimeIndices = np.argsort(childrenRuntime)
        n = 2 # number of top lists we want
        prevTopDropLists = []
        for i in range(n):
            # 0th should be last item, 1st should be second last, etc.
            topNIndex = sortedChildrenRuntimeIndices(sortedChildrenRuntimeIndices.shape[0]-1-n)
            prevTopDropLists.append(childrenDropLists[topNIndex])


def combineDropLists(Lists, length):
    #Lists = a list of lists
    if len(Lists) != 0:
        list1 = Lists[0]
        list2 = Lists[1]
        combinedList = []
        for i in range(length):
            x = np.random.random()
            if np.round(x) ==0:
                combinedList.append(list1[i])
            else:
                combinedList.append(list2[i])
        return combinedList
    else:
        return []
    """

if __name__ == '__main__':
    tf.app.run()
