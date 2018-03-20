"""
Uses a series of hyperparameter tuning methods in order to speed up FCN8

Based on train.py by Marvin Teichmann
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import random
import commentjson
import logging
import os
import glob
import sys
import string
import numpy as np

import collections


NUM_GENERATIONS = 4

layerList = ['conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
             'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1',
             'conv5_2', 'conv5_3', 'fc6', 'fc7', 'score_pool3', 
             'score_pool4']

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

def train_and_get_results(to_drop, hypes, encoder_path):
    ga_content = {'encoder_name': 'fcn8_vgg',
                  'encoder_path': encoder_path,
                  'drop': to_drop}

    with open(hypes['ga_data'], 'w') as f:
        commentjson.dump(ga_content, f)

    tf.reset_default_graph() 
    train.do_training(hypes)
    # thanks to https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder-using-python
    runs = glob.glob('RUNS/*')
    latest_dir = max(runs, key=os.path.getctime)
    log = os.path.join(latest_dir, 'output.log')
    with open(log) as f:
        lines = list(reversed(f.read().splitlines()))
    
    duration = float(lines[1].split(':')[-1].strip())
    ap = float(lines[2].split(':')[-1].strip())
    maxf1 = float(lines[4].split(':')[-1].strip())
    return duration, maxf1, ap 


def have_a_good_time(first, second):
    first_filt = [f for f in first if random.choice([True, False])]
    second_filt = [s for s in second if random.choice([True, False])]
    return list(set(first_filt + second_filt))


def go_it_alone(sadboi):
    changes = random.randint(1, 5)
    for i in range(changes):
        layer = random.choice(layerList)
        if layer in sadboi:
            sadboi.remove(layer)
        else:
            sadboi.append(layer)
    return sadboi


def run_genetic_algorithm(hypes, encoder_path):
    base_runtime, base_ap, base_maxf1 = train_and_get_results([], hypes, encoder_path)

    generation = [['fc6'], ['conv2_2'], ['conv3_2'], ['conv4_2'], ['conv5_2']]
    for i in range(NUM_GENERATIONS):
        durations = []
        maxf1s = []
        aps = []
        fitnesses = []
        for to_drop in generation:
            runtime, ap, maxf1 = train_and_get_results(to_drop, hypes, encoder_path)
            durations.append(runtime)
            aps.append(ap)
            maxf1s.append(maxf1)
            fitness = maxf1 if runtime <= base_runtime else 1000000   # kinda a negative fitness, oh well
            fitnesses.append(fitness)
            logging.info('GENERATION: {}, DROP: {}, RUNTIME: {}, AP: {}, MAXF: {}'.format(i, to_drop, runtime, ap, maxf1))
        
        sorted_gen = sorted(generation, key=lambda x: fitnesses[generation.index(x)])
        top, second = sorted_gen[:2]

        if i < NUM_GENERATIONS - 1:
            generation = []
            generation.append(top)
            generation.append(have_a_good_time(top, second))
            generation.append(go_it_alone(top))
            generation.append(go_it_alone(top))
            generation.append(go_it_alone(second))

    logging.info('FINAL GENERATION: {}, DURATIONS: {}, MAXF1: {}, AP: {}'.format(generation, durations, maxf1s, aps))
    logging.info('we are DONEEE!!')


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
  
    encoder_path = hypes['model']['architecture_file']
    hypes['model']['architecture_file'] = '../encoder/stub.py'
    hypes['ga_data'] = 'ga_data.json'

    run_genetic_algorithm(hypes, encoder_path)

if __name__ == '__main__':
    tf.app.run()
